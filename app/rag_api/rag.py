import os
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ragpassword")

TOP_K = int(os.getenv("TOP_K", "4"))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model

def get_conn():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        cursor_factory=RealDictCursor,
    )

def embed_query(text: str) -> List[float]:
    model = get_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()

def retrieve(question: str, doc_group: Optional[str] = None) -> List[Dict[str, Any]]:
    qvec = embed_query(question)

    sql = """
    SELECT
      doc_title, doc_group, section, page, content,
      1 - (embedding <=> %s::vector) AS score
    FROM rag_chunks
    WHERE embedding IS NOT NULL
      AND (%s IS NULL OR doc_group = %s)
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (qvec, doc_group, doc_group, qvec, TOP_K))
            return cur.fetchall()
