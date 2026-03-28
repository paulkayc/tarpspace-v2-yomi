import os
import time
import json
import uuid
import logging
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "google/gemini-2.5-flash"
TOP_K = 8
ACTIVITY_THRESHOLD = 0.3

EXTRACTION_SYSTEM = """You are a query parser. Extract location and activity from the user query.
Return only a JSON object with exactly two fields:
  - location: the city or place mentioned, or null if none
  - activity: what kind of person or skill they are looking for, or null if none
Example: {"location": "Houston", "activity": "guitar teacher"}
Return only valid JSON. No markdown. No explanation."""

VALIDATION_SYSTEM = """You are a matching validator for a social connection platform.
Given a user query and a list of candidate profiles, evaluate each one.
Return a JSON array. Each element must have:
  - id: the profile id
  - match: true or false
  - score: float 0.0 to 1.0
  - reason: one sentence
  - caveat: one sentence or null
Return only valid JSON. No markdown. No explanation."""


class TarpSpaceMatcher:
    def __init__(self, db_session):
        log.info(f"Loading embedding model: {EMBED_MODEL}")
        self.model = SentenceTransformer(EMBED_MODEL)
        self.db = db_session
        self._seed_if_empty()

    def _seed_if_empty(self):
        from db.database import seed_inventory
        seed_inventory(self.db, model=self.model)

    def _embed(self, text):
        emb = self.model.encode([text], normalize_embeddings=True)
        return emb[0].tolist()

    def _extract_query_fields(self, query):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return {"location": None, "activity": None}
        try:
            resp = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": EXTRACTION_SYSTEM},
                        {"role": "user", "content": query}
                    ]
                },
                timeout=15
            )
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            return json.loads(raw)
        except Exception as e:
            log.warning(f"Query extraction failed: {e}")
            return {"location": None, "activity": None}

    def _query_database(self, query_vec, activity_vec, location, top_k, use_activity):
        query_vec_str = str(query_vec)
        activity_vec_str = str(activity_vec)

        if use_activity and location:
            sql = text("""
                SELECT id, name, intent_type, activity, about, location_raw,
                       about_embedding <=> :query_vec AS about_score,
                       activity_embedding <=> :activity_vec AS activity_score
                FROM inventory
                WHERE location_raw ILIKE :loc
                AND activity_embedding <=> :activity_vec < :threshold
                ORDER BY about_embedding <=> :query_vec
                LIMIT :top_k
            """)
            params = {
                "query_vec": query_vec_str,
                "activity_vec": activity_vec_str,
                "loc": f"%{location}%",
                "threshold": ACTIVITY_THRESHOLD,
                "top_k": top_k
            }

        elif use_activity and not location:
            sql = text("""
                SELECT id, name, intent_type, activity, about, location_raw,
                       about_embedding <=> :query_vec AS about_score,
                       activity_embedding <=> :activity_vec AS activity_score
                FROM inventory
                WHERE activity_embedding <=> :activity_vec < :threshold
                ORDER BY about_embedding <=> :query_vec
                LIMIT :top_k
            """)
            params = {
                "query_vec": query_vec_str,
                "activity_vec": activity_vec_str,
                "threshold": ACTIVITY_THRESHOLD,
                "top_k": top_k
            }

        elif location and not use_activity:
            sql = text("""
                SELECT id, name, intent_type, activity, about, location_raw,
                       about_embedding <=> :query_vec AS about_score,
                       1.0 AS activity_score
                FROM inventory
                WHERE location_raw ILIKE :loc
                ORDER BY about_embedding <=> :query_vec
                LIMIT :top_k
            """)
            params = {
                "query_vec": query_vec_str,
                "loc": f"%{location}%",
                "top_k": top_k
            }

        else:
            sql = text("""
                SELECT id, name, intent_type, activity, about, location_raw,
                       about_embedding <=> :query_vec AS about_score,
                       1.0 AS activity_score
                FROM inventory
                ORDER BY about_embedding <=> :query_vec
                LIMIT :top_k
            """)
            params = {
                "query_vec": query_vec_str,
                "top_k": top_k
            }

        rows = self.db.execute(sql, params).fetchall()

        return [
            {
                "id": str(r[0]),
                "name": r[1],
                "intent_type": r[2],
                "activity": r[3],
                "about": r[4],
                "location_raw": r[5],
                "about_score": float(r[6]),
                "activity_score": float(r[7]),
            }
            for r in rows
        ]

    def _validate_with_llm(self, query, candidates):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None

        candidate_text = "\n".join([
            f"ID: {c['id']} | Name: {c['name']} | Activity: {c['activity']} | About: {c['about']} | Location: {c['location_raw']}"
            for c in candidates
        ])

        user_msg = f"User query: {query}\n\nCandidates:\n{candidate_text}"

        try:
            resp = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": VALIDATION_SYSTEM},
                        {"role": "user", "content": user_msg}
                    ]
                },
                timeout=30
            )
            response_json = resp.json()
            if "error" in response_json:
                log.error(f"OpenRouter error: {response_json['error']}")
                return None
            raw = response_json["choices"][0]["message"]["content"].strip()
            if not raw:
                log.error("LLM returned empty response")
                return None
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            log.error(f"LLM validation failed: {e}")
            log.error(f"Raw response: {resp.text[:500] if resp else 'no response'}")
            return None

    def _save_search_run(self, query, result_count, latency_ms, mandate_id=None):
        run_id = str(uuid.uuid4())
        safe_mandate_id = None
        if mandate_id:
            try:
                uuid.UUID(str(mandate_id))
                safe_mandate_id = mandate_id
            except ValueError:
                safe_mandate_id = None
        try:
            self.db.execute(text("""
                INSERT INTO search_runs (id, mandate_id, query, result_count, latency_ms)
                VALUES (:id, :mandate_id, :query, :result_count, :latency_ms)
            """), {
                "id": run_id,
                "mandate_id": safe_mandate_id,
                "query": query,
                "result_count": result_count,
                "latency_ms": latency_ms
            })
            self.db.commit()
        except Exception as e:
            log.warning(f"Could not save search run: {e}")
            self.db.rollback()
        return run_id

    def _save_search_results(self, search_run_id, results):
        for r in results:
            try:
                self.db.execute(text("""
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
                    "similarity_score": r.get("about_score"),
                    "alignment_score": r.get("score"),
                    "match": r.get("match", False),
                    "explanation": r.get("reason"),
                    "caveat": r.get("caveat")
                })
            except Exception as e:
                log.warning(f"Could not save result: {e}")
        try:
            self.db.commit()
        except Exception as e:
            log.warning(f"Could not commit results: {e}")
            self.db.rollback()

    def match(self, query, top_k=TOP_K, mandate_id=None, mandate=None, owner_id=None, use_activity=False):
        t0 = time.time()

        extracted = self._extract_query_fields(query)
        location = extracted.get("location")
        activity = extracted.get("activity")

        log.info(f"Extracted — location: {location}, activity: {activity}")

        query_vec = self._embed(query)
        activity_vec = self._embed(activity) if activity else query_vec

        candidates = self._query_database(query_vec, activity_vec, location, top_k, use_activity)

        log.info(f"Database returned {len(candidates)} candidates")

        if not candidates:
            return {
                "query": query,
                "extracted": extracted,
                "latency_ms": int((time.time() - t0) * 1000),
                "approach": "activity+about" if use_activity else "about_only",
                "results": []
            }

        validated = self._validate_with_llm(query, candidates)

        results = []
        if validated:
            for item in validated:
                candidate = next((c for c in candidates if c["id"] == item["id"]), {})
                item["about_score"] = candidate.get("about_score", 0)
                results.append(item)
            results.sort(key=lambda x: x.get("score", 0), reverse=True)
        else:
            results = candidates

        latency_ms = int((time.time() - t0) * 1000)
        run_id = self._save_search_run(query, len(results), latency_ms, mandate_id)
        self._save_search_results(run_id, results)

        return {
            "query": query,
            "extracted": extracted,
            "latency_ms": latency_ms,
            "approach": "activity+about" if use_activity else "about_only",
            "location_filter": location,
            "results": results
        }
