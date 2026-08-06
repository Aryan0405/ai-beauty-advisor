"""Request and response contracts for recommendation endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    """Optional exact-match filters for a recommendation search."""

    category: str | None = None
    brand: str | None = None
    skin_type: str | None = None

    def active_values(self) -> dict[str, str]:
        """Return only filters that have a non-empty value."""
        return {
            name: value
            for name, value in self.model_dump().items()
            if value is not None and value.strip()
        }


class SearchRequest(BaseModel):
    """A natural-language recommendation request."""

    query: str = Field(min_length=1, max_length=500)
    filters: SearchFilters | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class ProductResponse(BaseModel):
    """Public product metadata returned from the API."""

    product_id: str
    name: str
    brand: str
    category: str
    skin_type: str
    ingredients: str
    description: str
    price: float | None
    source_rank: float


class RecommendationResponse(ProductResponse):
    """Product metadata plus semantic and hybrid ranking scores."""

    similarity_score: float
    match_score: float


class SearchResponse(BaseModel):
    """Recommendation results and their optional Gemini explanation."""

    query: str
    count: int
    recommendations: list[RecommendationResponse]
    explanation: str


class HealthResponse(BaseModel):
    """Minimal response for deployment health checks."""

    status: str
