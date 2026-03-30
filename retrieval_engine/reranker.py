import os
import json
import logging
import requests

log = logging.getLogger(__name__)

LLM_MODEL = "google/gemini-2.5-flash"

VALIDATION_SYSTEM = """You are a matching validator for a social connection platform.
Given a user query and a list of candidate profiles, evaluate each one.
Return a JSON array. Each element must have:
  - id: the profile id
  - match: true or false
  - score: float 0.0 to 1.0
  - reasons: array of short strings explaining why
  - caveat: one sentence or null
Return only valid JSON. No markdown. No explanation."""


def rerank(query_text: str, candidates: list) -> tuple:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key or not candidates:
        log.warning("Skipping LLM reranking — no API key or no candidates")
        return _fallback_results(candidates), "vector_only", 0.0

    candidate_text = "\n".join([
        f"ID: {c['id']} | Name: {c['name']} | Activity: {c['activity']} | About: {c['about']} | Location: {c['location_raw']}"
        for c in candidates
    ])

    user_msg = f"User query: {query_text}\n\nCandidates:\n{candidate_text}"

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
            return _fallback_results(candidates), "vector_only", 0.0

        raw = response_json["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        if not raw:
            log.error("LLM returned empty response")
            return _fallback_results(candidates), "vector_only", 0.0

        validated = json.loads(raw)

        candidate_map = {c["id"]: c for c in candidates}
        results = []
        scores = []

        for item in validated:
            candidate = candidate_map.get(item["id"], {})
            score = float(item.get("score", 0.0))
            scores.append(score)

            reasons = item.get("reasons", [])
            if isinstance(reasons, str):
                reasons = [reasons]

            results.append({
                "id": item["id"],
                "name": candidate.get("name"),
                "activity": candidate.get("activity"),
                "about": candidate.get("about"),
                "location_raw": candidate.get("location_raw"),
                "score": score,
                "vector_score": candidate.get("vector_score", 0.0),
                "rerank_score": score,
                "match": item.get("match", False),
                "reasons": reasons,
                "caveat": item.get("caveat"),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        confidence = sum(scores) / len(scores) if scores else 0.0

        log.info(f"Reranking complete — confidence: {confidence:.2f}")
        return results, "vector+rules+llm", round(confidence, 2)

    except Exception as e:
        log.error(f"LLM reranking failed: {e}")
        return _fallback_results(candidates), "vector_only", 0.0


def _fallback_results(candidates: list) -> list:
    return [
        {
            "id": c["id"],
            "name": c.get("name"),
            "activity": c.get("activity"),
            "about": c.get("about"),
            "location_raw": c.get("location_raw"),
            "score": 1.0 - c.get("vector_score", 0.5),
            "vector_score": c.get("vector_score", 0.0),
            "rerank_score": 0.0,
            "match": None,
            "reasons": ["vector similarity match"],
            "caveat": "LLM reranking unavailable — vector only results",
        }
        for c in candidates
    ]
