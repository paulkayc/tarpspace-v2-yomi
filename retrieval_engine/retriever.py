import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

log = logging.getLogger(__name__)

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
ACTIVITY_THRESHOLD = 0.3

_model = None


def get_model():
    global _model
    if _model is None:
        log.info(f"Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(text_input: str) -> list:
    model = get_model()
    emb = model.encode([text_input], normalize_embeddings=True)
    return emb[0].tolist()


def retrieve(db, query_text: str, location: str, activity: str, top_k: int) -> tuple:
    model = get_model()

    query_vec = embed(query_text)
    activity_vec = embed(activity) if activity else query_vec

    query_vec_str = str(query_vec)
    activity_vec_str = str(activity_vec)

    location = location.strip() if location and isinstance(location, str) else None

    if location and activity:
        sql = text("""
            SELECT id, name, intent_type, activity, about, location_raw,
                   about_embedding <=> :query_vec AS vector_score
            FROM inventory
            WHERE location_raw ILIKE :loc
            AND activity_embedding <=> :activity_vec < :threshold
            ORDER BY about_embedding <=> :query_vec ASC
            LIMIT :top_k
        """)
        params = {
            "query_vec": query_vec_str,
            "activity_vec": activity_vec_str,
            "loc": f"%{location}%",
            "threshold": ACTIVITY_THRESHOLD,
            "top_k": top_k
        }
        strategy = "vector+location+activity"

    elif location and not activity:
        sql = text("""
            SELECT id, name, intent_type, activity, about, location_raw,
                   about_embedding <=> :query_vec AS vector_score
            FROM inventory
            WHERE location_raw ILIKE :loc
            ORDER BY about_embedding <=> :query_vec ASC
            LIMIT :top_k
        """)
        params = {
            "query_vec": query_vec_str,
            "loc": f"%{location}%",
            "top_k": top_k
        }
        strategy = "vector+location"

    elif activity and not location:
        sql = text("""
            SELECT id, name, intent_type, activity, about, location_raw,
                   about_embedding <=> :query_vec AS vector_score
            FROM inventory
            WHERE activity_embedding <=> :activity_vec < :threshold
            ORDER BY about_embedding <=> :query_vec ASC
            LIMIT :top_k
        """)
        params = {
            "query_vec": query_vec_str,
            "activity_vec": activity_vec_str,
            "threshold": ACTIVITY_THRESHOLD,
            "top_k": top_k
        }
        strategy = "vector+activity"

    else:
        sql = text("""
            SELECT id, name, intent_type, activity, about, location_raw,
                   about_embedding <=> :query_vec AS vector_score
            FROM inventory
            ORDER BY about_embedding <=> :query_vec ASC
            LIMIT :top_k
        """)
        params = {
            "query_vec": query_vec_str,
            "top_k": top_k
        }
        strategy = "vector_only"

    rows = db.execute(sql, params).fetchall()
    log.info(f"Retrieved {len(rows)} candidates using strategy: {strategy}")

    candidates = [
        {
            "id": str(r[0]),
            "name": r[1],
            "intent_type": r[2],
            "activity": r[3],
            "about": r[4],
            "location_raw": r[5],
            "vector_score": float(r[6]),
        }
        for r in rows
    ]

    return candidates, strategy
