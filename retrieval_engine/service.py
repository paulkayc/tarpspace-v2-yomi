import time
import uuid
import logging
from sqlalchemy import text

from retrieval_engine.parser import parse_query
from retrieval_engine.retriever import retrieve
from retrieval_engine.reranker import rerank
from retrieval_engine.schemas import (
    SearchRequest, SearchResponse, SearchResult,
    SearchDiagnostics, ParsedQuery,
    FeedbackRequest, FeedbackResponse
)

log = logging.getLogger(__name__)


def run_search(db, request: SearchRequest) -> SearchResponse:
    t0 = time.time()

    filters = request.filters.model_dump() if request.filters else {}
    parsed = parse_query(request.query_text, filters)

    location = parsed.get("location")
    activity = parsed.get("activity")

    candidates, strategy = retrieve(
        db=db,
        query_text=request.query_text,
        location=location,
        activity=activity,
        top_k=request.top_k
    )

    retrieved_k = len(candidates)

    if not candidates:
        log.warning("No candidates found")
        latency_ms = int((time.time() - t0) * 1000)
        run_id = _save_search_run(db, request, 0, latency_ms, strategy)
        return SearchResponse(
            search_run_id=run_id,
            request_id=request.request_id,
            query=request.query_text,
            parsed=ParsedQuery(location=location, activity=activity),
            strategy=strategy,
            confidence=0.0,
            results=[],
            diagnostics=SearchDiagnostics(
                latency_ms=latency_ms,
                retrieved_k=0,
                returned_k=0,
                strategy=strategy,
                location_filter=location,
                activity_filter=activity
            )
        )

    results, final_strategy, confidence = rerank(request.query_text, candidates)
    returned_k = len(results)
    latency_ms = int((time.time() - t0) * 1000)

    run_id = _save_search_run(db, request, returned_k, latency_ms, final_strategy)
    _save_search_results(db, run_id, results)

    search_results = [
        SearchResult(
            id=r["id"],
            name=r.get("name"),
            activity=r.get("activity"),
            about=r.get("about"),
            location_raw=r.get("location_raw"),
            score=r.get("score", 0.0),
            vector_score=r.get("vector_score", 0.0),
            rerank_score=r.get("rerank_score", 0.0),
            match=r.get("match"),
            reasons=r.get("reasons", []),
            caveat=r.get("caveat"),
        )
        for r in results
    ]

    return SearchResponse(
        search_run_id=run_id,
        request_id=request.request_id,
        query=request.query_text,
        parsed=ParsedQuery(location=location, activity=activity),
        strategy=final_strategy,
        confidence=confidence,
        results=search_results,
        diagnostics=SearchDiagnostics(
            latency_ms=latency_ms,
            retrieved_k=retrieved_k,
            returned_k=returned_k,
            strategy=final_strategy,
            location_filter=location,
            activity_filter=activity
        )
    )


def submit_feedback(db, request: FeedbackRequest) -> FeedbackResponse:
    feedback_id = str(uuid.uuid4())
    try:
        db.execute(text("""
            INSERT INTO search_feedback (
                id, search_run_id, result_id, label, user_id, notes, created_at
            ) VALUES (
                :id, :search_run_id, :result_id, :label, :user_id, :notes, NOW()
            )
        """), {
            "id": feedback_id,
            "search_run_id": request.search_run_id,
            "result_id": request.result_id,
            "label": request.label,
            "user_id": request.user_id,
            "notes": request.notes,
        })
        db.commit()
        return FeedbackResponse(feedback_id=feedback_id, status="saved")
    except Exception as e:
        log.error(f"Failed to save feedback: {e}")
        db.rollback()
        return FeedbackResponse(feedback_id=feedback_id, status="failed")


def _save_search_run(db, request: SearchRequest, result_count: int, latency_ms: int, strategy: str) -> str:
    run_id = str(uuid.uuid4())
    safe_mandate_id = None
    if request.mandate_id:
        try:
            import uuid as uuid_mod
            uuid_mod.UUID(str(request.mandate_id))
            safe_mandate_id = request.mandate_id
        except ValueError:
            safe_mandate_id = None

    safe_user_id = None
    if request.user_id:
        try:
            import uuid as uuid_mod
            uuid_mod.UUID(str(request.user_id))
            safe_user_id = request.user_id
        except ValueError:
            safe_user_id = None

    try:
        db.execute(text("""
            INSERT INTO search_runs (
                id, mandate_id, query, result_count, latency_ms, created_at
            ) VALUES (
                :id, :mandate_id, :query, :result_count, :latency_ms, NOW()
            )
        """), {
            "id": run_id,
            "mandate_id": safe_mandate_id,
            "query": request.query_text,
            "result_count": result_count,
            "latency_ms": latency_ms,
        })
        db.commit()
    except Exception as e:
        log.warning(f"Could not save search run: {e}")
        db.rollback()
    return run_id


def _save_search_results(db, search_run_id: str, results: list):
    for r in results:
        try:
            db.execute(text("""
                INSERT INTO search_results (
                    id, search_run_id, inventory_id,
                    similarity_score, alignment_score,
                    match, explanation, caveat
                ) VALUES (
                    gen_random_uuid(), :run_id, :inventory_id,
                    :similarity_score, :alignment_score,
                    :match, :explanation, :caveat
                )
            """), {
                "run_id": search_run_id,
                "inventory_id": r.get("id"),
                "similarity_score": r.get("vector_score"),
                "alignment_score": r.get("rerank_score"),
                "match": r.get("match", False),
                "explanation": " | ".join(r.get("reasons", [])),
                "caveat": r.get("caveat"),
            })
        except Exception as e:
            log.warning(f"Could not save result: {e}")
    try:
        db.commit()
    except Exception as e:
        log.warning(f"Could not commit results: {e}")
        db.rollback()
