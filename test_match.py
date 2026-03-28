import os
import sys
from dotenv import load_dotenv

load_dotenv()

from db.database import SessionLocal
from core.matcher import TarpSpaceMatcher

DEMO_QUERIES = [
    "I need someone to teach me guitar in Houston",
    "I want to learn coding in Lagos",
    "Looking for a personal trainer in Houston",
    "I want to connect with musicians in Lagos",
    "I need a chef to teach me cooking in Houston",
]


def run_query(query, matcher):
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    result = matcher.match(query)
    print(f"Latency: {result['latency_ms']}ms")
    print(f"Extracted: location={result['extracted'].get('location')} | activity={result['extracted'].get('activity')}")
    print(f"{'-'*70}")

    has_validation = any(r.get("match") is not None for r in result["results"])

    if has_validation:
        matched = [r for r in result["results"] if r.get("match")]
        not_matched = [r for r in result["results"] if not r.get("match")]

        print(f"MATCHES ({len(matched)}):")
        for r in matched:
            print(f"  [{r['id'][:8]}] {r.get('name', '')} | score: {r.get('score', 0):.2f}")
            print(f"  {r.get('reason', '')}")
            if r.get("caveat"):
                print(f"  Note: {r['caveat']}")

        if not_matched:
            print(f"\nNOT A MATCH ({len(not_matched)}):")
            for r in not_matched:
                print(f"  [{r['id'][:8]}] {r.get('name', '')} | {r.get('reason', '')}")
    else:
        print("VECTOR-ONLY (no LLM validation):")
        for r in result["results"]:
            print(f"  [{r['id'][:8]}] {r.get('name', '')} | {r.get('activity', '')} | {r.get('location_raw', '')}")

    print(f"{'='*70}")


def main():
    db = SessionLocal()
    try:
        print("Loading matcher...")
        matcher = TarpSpaceMatcher(db)

        if "--demo" in sys.argv:
            for q in DEMO_QUERIES:
                run_query(q, matcher)
        elif len(sys.argv) > 1:
            query = " ".join(a for a in sys.argv[1:] if not a.startswith("--"))
            if query:
                run_query(query, matcher)
        else:
            print("Usage:")
            print('  python test_match.py "your query"')
            print('  python test_match.py --demo')
    finally:
        db.close()


if __name__ == "__main__":
    main()
