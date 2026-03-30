import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

from db.database import get_db, seed_inventory
from retrieval_engine.schemas import (
    SearchRequest, SearchResponse,
    FeedbackRequest, FeedbackResponse
)
from retrieval_engine.service import run_search, submit_feedback
from retrieval_engine.retriever import get_model
from core.auth import verify_token, get_user_id

app = FastAPI(title="TarpSpace Retrieval Engine", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    from db.database import SessionLocal
    db = SessionLocal()
    try:
        model = get_model()
        seed_inventory(db, model=model)
    finally:
        db.close()


@app.get("/v1/health")
@app.get("/health")
def health():
    return {"status": "ok", "service": "TarpSpace Retrieval Engine", "version": "1.0.0"}


@app.post("/v1/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    db: Session = Depends(get_db),
    payload: dict = Depends(verify_token)
):
    user_id = get_user_id(payload)
    if not request.user_id:
        request.user_id = user_id
    if not request.query_text.strip():
        raise HTTPException(status_code=400, detail="query_text cannot be empty")
    return run_search(db, request)


@app.post("/v1/feedback", response_model=FeedbackResponse)
def feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db),
    payload: dict = Depends(verify_token)
):
    user_id = get_user_id(payload)
    if not request.user_id:
        request.user_id = user_id
    return submit_feedback(db, request)


@app.get("/v1/search-runs/{run_id}")
def get_search_run(
    run_id: str,
    db: Session = Depends(get_db),
    payload: dict = Depends(verify_token)
):
    row = db.execute(text(
        "SELECT id, query, result_count, latency_ms, created_at FROM search_runs WHERE id = :id"
    ), {"id": run_id}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Search run not found")
    results = db.execute(text(
        "SELECT id, inventory_id, similarity_score, alignment_score, match, explanation, caveat FROM search_results WHERE search_run_id = :id"
    ), {"id": run_id}).fetchall()
    return {
        "search_run_id": str(row[0]),
        "query": row[1],
        "result_count": row[2],
        "latency_ms": row[3],
        "created_at": str(row[4]),
        "results": [
            {
                "id": str(r[0]),
                "inventory_id": str(r[1]),
                "similarity_score": float(r[2]) if r[2] else None,
                "alignment_score": float(r[3]) if r[3] else None,
                "match": r[4],
                "explanation": r[5],
                "caveat": r[6],
            }
            for r in results
        ]
    }


@app.post("/v1/index/rebuild")
def rebuild_index(
    db: Session = Depends(get_db),
    payload: dict = Depends(verify_token)
):
    from retrieval_engine.retriever import get_model
    from db.database import seed_inventory
    model = get_model()
    db.execute(text("TRUNCATE TABLE inventory"))
    db.commit()
    seed_inventory(db, model=model)
    return {"status": "rebuilt", "message": "Inventory index has been rebuilt"}


@app.post("/match")
def match_legacy(
    request: SearchRequest,
    db: Session = Depends(get_db),
    payload: dict = Depends(verify_token)
):
    return run_search(db, request)


@app.get("/profiles")
def list_profiles(
    db: Session = Depends(get_db),
    payload: dict = Depends(verify_token)
):
    rows = db.execute(text(
        "SELECT id, name, activity, about, location_raw, is_active FROM inventory ORDER BY created_at"
    )).fetchall()
    return [
        {"id": str(r[0]), "name": r[1], "activity": r[2], "about": r[3], "location_raw": r[4], "is_active": r[5]}
        for r in rows
    ]


@app.get("/profiles/{profile_id}")
def get_profile(
    profile_id: str,
    db: Session = Depends(get_db),
    payload: dict = Depends(verify_token)
):
    row = db.execute(text(
        "SELECT id, name, activity, about, location_raw, is_active FROM inventory WHERE id = :id"
    ), {"id": profile_id}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Profile not found")
    return {"id": str(row[0]), "name": row[1], "activity": row[2], "about": row[3], "location_raw": row[4], "is_active": row[5]}


@app.get("/logs")
def get_logs(
    db: Session = Depends(get_db),
    payload: dict = Depends(verify_token)
):
    rows = db.execute(text(
        "SELECT id, query, result_count, latency_ms, created_at FROM search_runs ORDER BY created_at DESC LIMIT 50"
    )).fetchall()
    return [
        {"id": str(r[0]), "query": r[1], "result_count": r[2], "latency_ms": r[3], "created_at": str(r[4])}
        for r in rows
    ]


@app.get("/me")
def get_me(payload: dict = Depends(verify_token)):
    return {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
        "role": payload.get("role")
    }