from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime
import uuid


class SearchFilters(BaseModel):
    location: Optional[str] = None
    activity: Optional[str] = None


class SearchRequest(BaseModel):
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    mandate_id: Optional[str] = None
    query_text: str
    intent: Optional[dict] = None
    filters: Optional[SearchFilters] = None
    top_k: int = 8


class SearchResult(BaseModel):
    id: str
    name: Optional[str] = None
    activity: Optional[str] = None
    about: Optional[str] = None
    location_raw: Optional[str] = None
    score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float = 0.0
    match: Optional[bool] = None
    reasons: List[str] = []
    caveat: Optional[str] = None


class SearchDiagnostics(BaseModel):
    latency_ms: int
    retrieved_k: int
    returned_k: int
    strategy: str
    location_filter: Optional[str] = None
    activity_filter: Optional[str] = None


class ParsedQuery(BaseModel):
    location: Optional[str] = None
    activity: Optional[str] = None


class SearchResponse(BaseModel):
    search_run_id: str
    request_id: Optional[str] = None
    query: str
    parsed: ParsedQuery
    strategy: str
    confidence: float
    results: List[SearchResult]
    diagnostics: SearchDiagnostics


class FeedbackRequest(BaseModel):
    search_run_id: str
    result_id: str
    label: str
    user_id: Optional[str] = None
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str
