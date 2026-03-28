import os
import logging
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

log = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_inventory(db, model=None):
    result = db.execute(text("SELECT COUNT(*) FROM inventory")).scalar()
    if result > 0:
        log.info(f"Inventory already has {result} agents — skipping seed")
        return

    from data.agents import AGENTS
    log.info(f"Seeding {len(AGENTS)} agents into inventory...")

    if model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model for seeding...")
        model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    about_texts = [a["about"] or "" for a in AGENTS]
    activity_texts = [f"{a['activity'] or ''} {a['about'] or ''}" for a in AGENTS]

    log.info("Computing about embeddings...")
    about_embeddings = model.encode(about_texts, normalize_embeddings=True, show_progress_bar=True)

    log.info("Computing activity embeddings...")
    activity_embeddings = model.encode(activity_texts, normalize_embeddings=True, show_progress_bar=True)

    log.info("Inserting agents into database...")
    for i, agent in enumerate(AGENTS):
        about_vec = about_embeddings[i].tolist()
        activity_vec = activity_embeddings[i].tolist()

        db.execute(text("""
            INSERT INTO inventory (
                id, name, intent_type, activity, about,
                location_raw, is_active,
                about_embedding, activity_embedding
            ) VALUES (
                gen_random_uuid(),
                :name,
                :intent_type,
                :activity,
                :about,
                :location_raw,
                true,
                :about_embedding,
                :activity_embedding
            )
        """), {
            "name": agent["name"],
            "intent_type": agent["intent_type"],
            "activity": agent["activity"],
            "about": agent["about"],
            "location_raw": agent["location_raw"],
            "about_embedding": str(about_vec),
            "activity_embedding": str(activity_vec),
        })

    db.commit()
    log.info("Seeding complete — embeddings stored in database")


def log_activity(db, owner_id, mandate_id, event_type, payload):
    import json
    db.execute(text("""
        INSERT INTO activity_log (owner_id, mandate_id, event_type, payload)
        VALUES (:owner_id, :mandate_id, :event_type, :payload::jsonb)
    """), {
        "owner_id": owner_id,
        "mandate_id": mandate_id,
        "event_type": event_type,
        "payload": json.dumps(payload)
    })
    db.commit()
