"""Precision@5 relevance spot-check (spec section 19).

A fully rigorous relevance evaluation needs a human judgment per
query-result pair, which is out of scope for this project. This script
instead runs a *heuristic* spot-check: each curated query in
relevance_cases.py is paired with the category/skin-type signals a
reasonable result should have -- decided by hand, i.e. a human ("manual",
in the spec's sense) chose what counts as relevant for each query -- and a
result is scored as relevant if it matches those signals. This makes the
spec's "manual spot-check precision@5" repeatable and scriptable, but it is
a structural proxy for relevance, not a substitute for a human actually
reading the results and judging quality.

Relevance rule for one retrieved product, given a query's expected signals:
  - The product's category must be one of `expected_categories`.
  - If `expected_skin_types` is non-empty, the product's skin_type tags
    (comma-separated) must include at least one of them. Skin type is not
    checked when a query has no clear skin-type signal.

This runs against the real, committed SQLite DB and FAISS index (same as
run_latency_eval.py) -- not a test fixture -- so it needs `data/` to be
populated by the ingestion pipeline. It makes no Gemini calls.

Usage:
    python -m backend.tests.eval.run_precision_eval
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from backend.tests.eval.relevance_cases import RELEVANCE_CASES


DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent / "results" / "precision_eval.json"
LOW_PRECISION_WARNING_THRESHOLD = 0.6


@dataclass
class CaseResult:
    """One query's top-5 retrieval results and their relevance flags."""

    query: str
    expected_categories: list[str]
    expected_skin_types: list[str]
    retrieved: list[dict] = field(default_factory=list)
    relevant_flags: list[bool] = field(default_factory=list)

    @property
    def precision_at_5(self) -> float:
        if not self.relevant_flags:
            return 0.0
        return sum(self.relevant_flags) / len(self.relevant_flags)


def _is_relevant(
    product: dict, expected_categories: set[str], expected_skin_types: set[str]
) -> bool:
    if product["category"] not in expected_categories:
        return False
    if not expected_skin_types:
        return True
    product_skin_types = {tag.strip() for tag in product["skin_type"].split(",") if tag.strip()}
    return bool(product_skin_types & expected_skin_types)


def run_case(case: dict, top_k: int = 5) -> CaseResult:
    from backend.app.services.recommendation import get_recommendations

    expected_categories = set(case["expected_categories"])
    expected_skin_types = set(case.get("expected_skin_types", ()))
    result = CaseResult(
        query=case["query"],
        expected_categories=sorted(expected_categories),
        expected_skin_types=sorted(expected_skin_types),
    )

    recommendations = get_recommendations(case["query"], top_k=top_k)
    for recommendation in recommendations:
        product = recommendation.product
        product_dict = {
            "product_id": product.product_id,
            "name": product.name,
            "category": product.category,
            "skin_type": product.skin_type,
        }
        result.retrieved.append(product_dict)
        result.relevant_flags.append(
            _is_relevant(product_dict, expected_categories, expected_skin_types)
        )

    return result


def _print_case(case_result: CaseResult) -> None:
    hits = sum(case_result.relevant_flags)
    marker = "OK " if case_result.precision_at_5 >= LOW_PRECISION_WARNING_THRESHOLD else "LOW"
    print(f"[{marker}] {hits}/{len(case_result.relevant_flags)}  {case_result.query}")
    for product, is_relevant in zip(case_result.retrieved, case_result.relevant_flags):
        flag = "+" if is_relevant else "-"
        print(f"       {flag} {product['category']:<12} {product['name']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Heuristic precision@5 relevance spot-check.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    args = parser.parse_args(argv)

    print(
        f"Running precision@{args.top_k} spot-check over {len(RELEVANCE_CASES)} "
        "curated queries (heuristic: category + skin_type match against "
        "hand-authored expected signals)...\n"
    )

    case_results = [run_case(case, top_k=args.top_k) for case in RELEVANCE_CASES]
    for case_result in case_results:
        _print_case(case_result)

    overall_precision = sum(cr.precision_at_5 for cr in case_results) / len(case_results)
    low_precision_queries = [cr.query for cr in case_results if cr.precision_at_5 < LOW_PRECISION_WARNING_THRESHOLD]

    print(f"\nOverall precision@{args.top_k}: {overall_precision:.1%} across {len(case_results)} queries")
    if low_precision_queries:
        print(f"Below {LOW_PRECISION_WARNING_THRESHOLD:.0%} threshold ({len(low_precision_queries)}):")
        for query in low_precision_queries:
            print(f"  - {query}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_k": args.top_k,
        "method": (
            "heuristic: product category in expected_categories AND "
            "(no expected_skin_types OR skin_type tags intersect them)"
        ),
        "query_count": len(case_results),
        "overall_precision_at_5": round(overall_precision, 4),
        "cases": [
            {
                "query": cr.query,
                "expected_categories": cr.expected_categories,
                "expected_skin_types": cr.expected_skin_types,
                "precision_at_5": round(cr.precision_at_5, 4),
                "retrieved": [
                    {**product, "relevant": flag}
                    for product, flag in zip(cr.retrieved, cr.relevant_flags)
                ],
            }
            for cr in case_results
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
