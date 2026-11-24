"""SQLite persistence: persisted run events permit SSE reconnection and inspections."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from .config import settings, prepare_storage


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(v):
    return json.dumps(v, ensure_ascii=False)


def decode(v, default=None):
    if v is None:
        return default
    return json.loads(v)


@contextmanager
def connect():
    prepare_storage()
    db = sqlite3.connect(str(settings.data_dir / "app.sqlite3"), timeout=20)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=20000")
    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def initialize():
    with connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, title TEXT NOT NULL,
            source_type TEXT NOT NULL, document_path TEXT, status TEXT NOT NULL,
            parser TEXT, error TEXT, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            seq INTEGER NOT NULL, section TEXT NOT NULL, content TEXT NOT NULL,
            paragraph_refs TEXT NOT NULL, pages TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_chunks_paper ON chunks(paper_id);
        CREATE TABLE IF NOT EXISTS paper_tables (
            paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            table_number INTEGER NOT NULL, page INTEGER NOT NULL,
            headers TEXT NOT NULL, rows_json TEXT NOT NULL, caption TEXT NOT NULL DEFAULT '',
            PRIMARY KEY(paper_id,table_number,page)
        );
        CREATE INDEX IF NOT EXISTS ix_paper_tables_paper ON paper_tables(paper_id);
        CREATE TABLE IF NOT EXISTS vector_cache (
            cache_file TEXT NOT NULL, paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            PRIMARY KEY(cache_file,paper_id)
        );
        CREATE TABLE IF NOT EXISTS eval_cases (
            question_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
            split TEXT NOT NULL, question TEXT NOT NULL, answers_json TEXT NOT NULL,
            evidence_json TEXT NOT NULL, answer_type TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, title TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL, content TEXT NOT NULL, citations TEXT NOT NULL,
            run_id TEXT, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            status TEXT NOT NULL, mode TEXT NOT NULL, question TEXT NOT NULL, paper_ids TEXT NOT NULL,
            answer TEXT, citations TEXT, error TEXT, metrics TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL, completed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS run_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            event_type TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_events_run ON run_events(run_id,id);
        CREATE TABLE IF NOT EXISTS profile (
            owner_id TEXT PRIMARY KEY, language TEXT NOT NULL DEFAULT 'en',
            explanation_level TEXT NOT NULL DEFAULT 'beginner', domains TEXT NOT NULL DEFAULT '[]',
            goals TEXT NOT NULL DEFAULT '[]', memory_enabled INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, domain TEXT NOT NULL,
            topic TEXT NOT NULL, memory_type TEXT NOT NULL, content TEXT NOT NULL,
            mastery TEXT NOT NULL DEFAULT 'mentioned', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        """)
        db.execute("INSERT OR IGNORE INTO profile(owner_id) VALUES(?)", (settings.owner_id,))
        # A single-worker local process is supported; interrupted runs must not remain 'running'.
        db.execute("UPDATE runs SET status='failed', error='Backend restarted before run completed', completed_at=? WHERE status IN ('pending','running')", (now(),))
        db.execute("UPDATE papers SET status='failed', error='Backend restarted before ingestion completed' WHERE status='processing'")


def rows(sql, args=()):
    with connect() as db:
        return [dict(r) for r in db.execute(sql, args).fetchall()]


def one(sql, args=()):
    with connect() as db:
        r = db.execute(sql, args).fetchone()
        return dict(r) if r else None


def execute(sql, args=()):
    with connect() as db:
        return db.execute(sql, args).rowcount


def event(run_id, event_type, payload):
    with connect() as db:
        db.execute("INSERT INTO run_events(run_id,event_type,payload,created_at) VALUES(?,?,?,?)",
                   (run_id, event_type, encode(payload), now()))
