"""Latency evaluation utility (spec sections 18-19).

Measures retrieval-only latency (embed query -> FAISS search -> SQLite
fetch -> hybrid ranking, i.e. everything the NFR table calls "vector search
alone (excluding LLM call)") against a P95 budget of 150ms for K=5, and can
optionally measure the full search+explanation pipeline -- including a real
Gemini call -- against a P95 budget of 3.5s.

This is a manual/QA tool (spec section 18's "Manual/QA" row: "Full
docker-compose run with real dataset and real Gemini key before each
release; verify P95 latency target"), not a pytest test. It needs the real
embedding model, the real FAISS index, and the real SQLite DB to produce a
meaningful number, and --with-llm makes a real network call to Gemini. It
is deliberately not named test_*.py so pytest never collects it, keeping
the offline test suite (spec section 18's CI gate) free of network calls.

Usage:
    python -m backend.tests.eval.run_latency_eval
    python -m backend.tests.eval.run_latency_eval --with-llm --top-k 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from backend.tests.eval.queries import EVAL_QUERIES


RETRIEVAL_P95_BUDGET_MS = 150.0
FULL_REQUEST_P95_BUDGET_MS = 3500.0
DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent / "results" / "latency_eval.json"


@dataclass
class LatencyStats:
    """Timing samples (milliseconds) for one measured phase."""

    label: str
    budget_ms: float
    samples_ms: list[float] = field(default_factory=list)

    @property
    def p50(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p95(self) -> float:
        if len(self.samples_ms) == 1:
            return self.samples_ms[0]
        return statistics.quantiles(self.samples_ms, n=100, method="inclusive")[93]

    @property
    def passed(self) -> bool:
        return self.p95 <= self.budget_ms

    def summary(self) -> dict:
        return {
            "label": self.label,
            "budget_ms": self.budget_ms,
            "count": len(self.samples_ms),
            "min_ms": round(min(self.samples_ms), 2),
            "mean_ms": round(statistics.mean(self.samples_ms), 2),
            "p50_ms": round(self.p50, 2),
            "p95_ms": round(self.p95, 2),
            "max_ms": round(max(self.samples_ms), 2),
            "passed": self.passed,
        }


def run_retrieval_eval(top_k: int, queries: list[str]) -> LatencyStats:
    """Time the retrieval-only path: embed + FAISS search + DB fetch + ranking."""
    from backend.app.services.recommendation import get_recommendations

    stats = LatencyStats(
        label="retrieval_only (embed + FAISS + DB + ranking, excludes LLM)",
        budget_ms=RETRIEVAL_P95_BUDGET_MS,
    )

    # Warm up outside the timed loop: the NFR is about steady-state request
    # latency, not one-time embedding-model / FAISS-index load cost.
    get_recommendations(queries[0], top_k=top_k)

    for query in queries:
        start = time.perf_counter()
        get_recommendations(query, top_k=top_k)
        stats.samples_ms.append((time.perf_counter() - start) * 1000)

    return stats


def run_full_request_eval(top_k: int, queries: list[str]) -> tuple[LatencyStats, bool]:
    """Time the full search+explanation path via a real HTTP request.

    Returns the stats plus whether every response actually got at least one
    non-null explanation -- if Gemini is unreachable or misconfigured, the
    request still succeeds (graceful degradation), but the measured time
    would understate real end-to-end latency.
    """
    from fastapi.testclient import TestClient

    from backend.app.main import app

    stats = LatencyStats(
        label="search_and_explanation (includes a live Gemini call)",
        budget_ms=FULL_REQUEST_P95_BUDGET_MS,
    )
    any_explanation_missing = False

    with TestClient(app) as client:
        client.post("/api/v1/recommend", json={"query": queries[0], "top_k": top_k})  # warm-up

        for query in queries:
            start = time.perf_counter()
            response = client.post("/api/v1/recommend", json={"query": query, "top_k": top_k})
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.raise_for_status()
            stats.samples_ms.append(elapsed_ms)

            recommendations = response.json().get("recommendations", [])
            if recommendations and not any(r.get("explanation") for r in recommendations):
                any_explanation_missing = True

    return stats, any_explanation_missing


def _print_summary(stats: LatencyStats) -> None:
    summary = stats.summary()
    verdict = "PASS" if summary["passed"] else "FAIL"
    print(f"\n[{verdict}] {summary['label']}")
    print(f"  budget (P95): <= {summary['budget_ms']:.0f}ms")
    print(f"  samples:      {summary['count']}")
    print(f"  min / mean:   {summary['min_ms']:.1f}ms / {summary['mean_ms']:.1f}ms")
    print(f"  P50 / P95:    {summary['p50_ms']:.1f}ms / {summary['p95_ms']:.1f}ms")
    print(f"  max:          {summary['max_ms']:.1f}ms")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate search latency against spec NFR targets.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help=(
            "Also measure the full search+explanation pipeline. Makes real "
            "Gemini API calls -- requires a working GEMINI_API_KEY and "
            "network access. Off by default so this stays a manual/QA "
            "tool, never a CI dependency."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args(argv)

    print(
        f"Running latency evaluation over {len(EVAL_QUERIES)} representative "
        f"queries (top_k={args.top_k})..."
    )

    all_passed = True
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_k": args.top_k,
        "query_count": len(EVAL_QUERIES),
        "phases": [],
    }

    retrieval_stats = run_retrieval_eval(args.top_k, EVAL_QUERIES)
    _print_summary(retrieval_stats)
    report["phases"].append(retrieval_stats.summary())
    all_passed &= retrieval_stats.passed

    if args.with_llm:
        full_stats, explanations_missing = run_full_request_eval(args.top_k, EVAL_QUERIES)
        _print_summary(full_stats)
        report["phases"].append(full_stats.summary())
        all_passed &= full_stats.passed
        if explanations_missing:
            print(
                "\nWARNING: at least one query got zero explanations back. "
                "Check GEMINI_API_KEY / network access -- this run's "
                "full-request timing may not reflect a real Gemini call."
            )
    else:
        print(
            "\nSkipped search_and_explanation phase (pass --with-llm to "
            "include it; it makes real Gemini API calls)."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.output}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
