"""Curated representative queries for latency and relevance evaluation.

Referenced by spec section 19 ("manual spot-check precision@5 against a
curated set of ~20 representative queries"). The same set doubles as the
query workload for the latency evaluation script, since both need
realistic, varied natural-language input.
"""

from __future__ import annotations

EVAL_QUERIES: list[str] = [
    "lightweight moisturizer for oily acne-prone skin",
    "gentle cleanser for sensitive skin",
    "brightening vitamin c serum for dull skin",
    "hydrating serum for dry skin",
    "anti-aging night cream with retinol",
    "mineral sunscreen for sensitive skin",
    "oil-free sunscreen for oily skin",
    "clay mask for oily and acne-prone skin",
    "eye cream for dark circles and puffiness",
    "fragrance-free moisturizer for sensitive skin",
    "exfoliating treatment for dull, uneven skin tone",
    "niacinamide serum for large pores",
    "rich night cream for very dry skin",
    "affordable daily moisturizer for normal skin",
    "soothing cream for redness and irritation",
    "hyaluronic acid serum for hydration",
    "cleansing balm for removing makeup",
    "spot treatment for acne breakouts",
    "SPF 50 sunscreen for combination skin",
    "luxury anti-aging serum with peptides",
]
