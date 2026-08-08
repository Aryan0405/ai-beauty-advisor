"""Hand-authored relevance ground truth for the precision@5 spot-check.

Pairs each query from queries.EVAL_QUERIES with the category/skin-type
signals a relevant top-5 result should plausibly have. These signals were
authored by hand against the category label set actually present in the
ingested dataset (verified directly against data/beauty_advisor.db):
Moisturizer, Cleanser, Face Mask, Treatment, Eye cream, Sun protect. There
is no "Serum" category in the source data, and it turns out the source
data is not even consistent about where serums land: spot-checking actual
results showed products literally named "... Serum" filed under both
Treatment (e.g. "Vitamin C Ester Brightening Serum") and Moisturizer (e.g.
"Ultra Repair(R) Hydrating Serum"). Queries asking for a "serum" therefore
accept either category below -- not because the retrieval is expected to
be sloppy about it, but because the ground truth would otherwise be
penalizing the system for a labeling inconsistency in the source dataset
it has no way to know about.

This is a deliberately coarse, structural proxy for relevance (category +
skin type tags), not a semantic judgment of whether a product is *actually*
a good match for the query's stated concern -- see run_precision_eval.py
for exactly how a result is scored against these signals, and its module
docstring for why this proxy was chosen over a fully automated one.
"""

from __future__ import annotations

from backend.tests.eval.queries import EVAL_QUERIES


RELEVANCE_CASES: list[dict] = [
    {
        "query": EVAL_QUERIES[0],  # lightweight moisturizer for oily acne-prone skin
        "expected_categories": ["Moisturizer"],
        "expected_skin_types": ["Oily"],
    },
    {
        "query": EVAL_QUERIES[1],  # gentle cleanser for sensitive skin
        "expected_categories": ["Cleanser"],
        "expected_skin_types": ["Sensitive"],
    },
    {
        "query": EVAL_QUERIES[2],  # brightening vitamin c serum for dull skin
        "expected_categories": ["Treatment", "Moisturizer"],
        "expected_skin_types": [],
    },
    {
        "query": EVAL_QUERIES[3],  # hydrating serum for dry skin
        "expected_categories": ["Treatment", "Moisturizer"],
        "expected_skin_types": ["Dry"],
    },
    {
        "query": EVAL_QUERIES[4],  # anti-aging night cream with retinol
        "expected_categories": ["Moisturizer", "Treatment"],
        "expected_skin_types": [],
    },
    {
        "query": EVAL_QUERIES[5],  # mineral sunscreen for sensitive skin
        "expected_categories": ["Sun protect"],
        "expected_skin_types": ["Sensitive"],
    },
    {
        "query": EVAL_QUERIES[6],  # oil-free sunscreen for oily skin
        "expected_categories": ["Sun protect"],
        "expected_skin_types": ["Oily"],
    },
    {
        "query": EVAL_QUERIES[7],  # clay mask for oily and acne-prone skin
        "expected_categories": ["Face Mask"],
        "expected_skin_types": ["Oily"],
    },
    {
        "query": EVAL_QUERIES[8],  # eye cream for dark circles and puffiness
        "expected_categories": ["Eye cream"],
        "expected_skin_types": [],
    },
    {
        "query": EVAL_QUERIES[9],  # fragrance-free moisturizer for sensitive skin
        "expected_categories": ["Moisturizer"],
        "expected_skin_types": ["Sensitive"],
    },
    {
        "query": EVAL_QUERIES[10],  # exfoliating treatment for dull, uneven skin tone
        "expected_categories": ["Treatment"],
        "expected_skin_types": [],
    },
    {
        "query": EVAL_QUERIES[11],  # niacinamide serum for large pores
        "expected_categories": ["Treatment", "Moisturizer"],
        "expected_skin_types": [],
    },
    {
        "query": EVAL_QUERIES[12],  # rich night cream for very dry skin
        "expected_categories": ["Moisturizer"],
        "expected_skin_types": ["Dry"],
    },
    {
        "query": EVAL_QUERIES[13],  # affordable daily moisturizer for normal skin
        "expected_categories": ["Moisturizer"],
        "expected_skin_types": ["Normal"],
    },
    {
        "query": EVAL_QUERIES[14],  # soothing cream for redness and irritation
        "expected_categories": ["Moisturizer", "Treatment"],
        "expected_skin_types": ["Sensitive"],
    },
    {
        "query": EVAL_QUERIES[15],  # hyaluronic acid serum for hydration
        "expected_categories": ["Treatment", "Moisturizer"],
        "expected_skin_types": [],
    },
    {
        "query": EVAL_QUERIES[16],  # cleansing balm for removing makeup
        "expected_categories": ["Cleanser"],
        "expected_skin_types": [],
    },
    {
        "query": EVAL_QUERIES[17],  # spot treatment for acne breakouts
        "expected_categories": ["Treatment"],
        "expected_skin_types": ["Oily"],
    },
    {
        "query": EVAL_QUERIES[18],  # SPF 50 sunscreen for combination skin
        "expected_categories": ["Sun protect"],
        "expected_skin_types": ["Combination"],
    },
    {
        "query": EVAL_QUERIES[19],  # luxury anti-aging serum with peptides
        "expected_categories": ["Treatment", "Moisturizer"],
        "expected_skin_types": [],
    },
]

_case_queries = {case["query"] for case in RELEVANCE_CASES}
_missing = set(EVAL_QUERIES) - _case_queries
if _missing:
    raise ValueError(f"queries.py and relevance_cases.py have drifted: missing cases for {_missing}")
