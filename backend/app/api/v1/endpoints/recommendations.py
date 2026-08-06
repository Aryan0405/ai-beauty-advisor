"""Recommendation endpoint combining retrieval and Gemini explanation."""

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas import (
    RecommendationResponse,
    SearchRequest,
    SearchResponse,
)
from backend.app.services.explanation import generate_explanation
from backend.app.services.recommendation import get_recommendations


router = APIRouter()


def _to_response(recommendation: object) -> RecommendationResponse:
    """Convert a service recommendation into the public API response schema."""
    product = recommendation.product
    return RecommendationResponse(
        product_id=product.product_id,
        name=product.name,
        brand=product.brand,
        category=product.category,
        skin_type=product.skin_type,
        ingredients=product.ingredients,
        description=product.description,
        price=product.price,
        source_rank=product.source_rank,
        similarity_score=recommendation.similarity_score,
        match_score=recommendation.match_score,
    )


@router.post("/recommend", response_model=SearchResponse)
def recommend(request: SearchRequest) -> SearchResponse:
    """Return ranked products and a grounded Gemini explanation."""
    try:
        recommendations = get_recommendations(
            query=request.query,
            filters=request.filters.active_values() if request.filters else None,
            top_k=request.top_k,
        )
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service is unavailable.",
        ) from error

    result_models = [_to_response(item) for item in recommendations]
    explanation = generate_explanation(
        request.query,
        [result.model_dump() for result in result_models],
    )
    return SearchResponse(
        query=request.query,
        count=len(result_models),
        recommendations=result_models,
        explanation=explanation,
    )
