import os
import json
import logging
import requests

log = logging.getLogger(__name__)

LLM_MODEL = "google/gemini-2.5-flash"

EXTRACTION_SYSTEM = """You are a query parser for a social connection platform.
Extract location and activity from the user query.
Return only a JSON object with exactly two fields:
  - location: the city or place mentioned, or null if none
  - activity: what kind of person or skill they are looking for, or null if none
Example: {"location": "Lagos", "activity": "guitar teacher"}
Return only valid JSON. No markdown. No explanation."""


def parse_query(query_text: str, filters: dict = None) -> dict:
    location = None
    activity = None

    if filters:
        location = filters.get("location")
        activity = filters.get("activity")

    if location and activity:
        log.info(f"Using filters directly — location: {location}, activity: {activity}")
        return {"location": location, "activity": activity}

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        log.warning("No API key — skipping LLM extraction")
        return {"location": location, "activity": activity}

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
                    {"role": "user", "content": query_text}
                ]
            },
            timeout=15
        )
        response_json = resp.json()

        if "error" in response_json:
            log.error(f"OpenRouter error: {response_json['error']}")
            return {"location": location, "activity": activity}

        raw = response_json["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        extracted = json.loads(raw)

        if not location:
            location = extracted.get("location")
        if not activity:
            activity = extracted.get("activity")

        log.info(f"Extracted — location: {location}, activity: {activity}")
        return {"location": location, "activity": activity}

    except Exception as e:
        log.error(f"Query extraction failed: {e}")
        return {"location": location, "activity": activity}
