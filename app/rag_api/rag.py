import os
import uuid
import hashlib
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

from pypdf import PdfReader
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

DOC_ROOT = os.getenv("DOC_ROOT", "/data/raw_docs")

# -------------------------
# [NEW] Text sanitization helpers
# -------------------------
def _sanitize_text(s: str) -> str:
    """
    PDF text extraction sometimes yields invalid unicode surrogate characters like '\\udfb3'.
    Those crash UTF-8 encoding when inserting into DB.

    Strategy:
    - Remove surrogate range: U+D800 ~ U+DFFF
    - Remove NULL bytes
    - Normalize line breaks a bit
    """
    if s is None:
        return ""
    # --- START: sanitize text (surrogates + null) ---
    s = s.replace("\x00", " ")
    s = re.sub(r"[\ud800-\udfff]", "", s)   # remove surrogate code points
    # --- END: sanitize text (surrogates + null) ---
    return s

def _sha256_text(s: str) -> str:
    # --- START: ensure sanitize before hashing ---
    s = _sanitize_text(s)
    # --- END: ensure sanitize before hashing ---
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    chunks: List[str] = []
    # --- START: sanitize before chunking ---
    text = _sanitize_text(text or "")
    # --- END: sanitize before chunking ---
    text = (text or "").strip()
    if not text:
        return chunks
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks

def _extract_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(pdf_path)
    pages: List[Dict[str, Any]] = []
    for idx, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        # --- START: sanitize extracted text ---
        txt = _sanitize_text(txt).strip()
        # --- END: sanitize extracted text ---
        if txt:
            pages.append({"page": idx, "text": txt})
    return pages

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = get_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return vecs.tolist()

def ingest_pdfs_to_db(
    doc_group: str,
    limit: Optional[int] = None,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> Dict[str, Any]:
    """
    DOC_ROOT 아래 PDF를 읽어 rag_chunks에 insert.
    - 중복 방지: chunk_hash unique
    - embedding: vector(384)

    NOTE:
    - 일부 PDF가 깨져도 전체 ingest가 죽지 않도록 per-file try/except로 계속 진행
    - 실패한 PDF는 errors에 기록
    """
    root = Path(DOC_ROOT)
    if not root.exists():
        raise RuntimeError(f"DOC_ROOT not found: {DOC_ROOT}")

    pdfs = sorted([p for p in root.rglob("*.pdf")])
    if limit:
        pdfs = pdfs[:limit]

    docs_indexed = 0
    docs_failed = 0
    chunks_attempted = 0
    chunks_inserted = 0

    errors: List[Dict[str, Any]] = []

    insert_sql = """
    INSERT INTO rag_chunks
      (id, doc_title, doc_group, section, page, content, embedding, doc_path, doc_hash, chunk_hash)
    VALUES
      (%(id)s, %(doc_title)s, %(doc_group)s, %(section)s, %(page)s, %(content)s, %(embedding)s::vector,
       %(doc_path)s, %(doc_hash)s, %(chunk_hash)s)
    ON CONFLICT (chunk_hash) DO NOTHING;
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            # ✅ START: 전후 count diff로 inserted 계산
            cur.execute("SELECT COUNT(*) AS cnt FROM rag_chunks;")
            count_before = cur.fetchone()["cnt"]
            # ✅ END

            for pdf_path in pdfs:
                try:
                    pages = _extract_pdf_pages(str(pdf_path))
                    if not pages:
                        continue

                    doc_title = pdf_path.stem
                    full_text = "\n\n".join([p["text"] for p in pages])
                    doc_hash = _sha256_text(full_text)

                    # 1) chunk 생성
                    chunk_rows = []
                    for p in pages:
                        page_no = int(p["page"])
                        for chunk in _chunk_text(p["text"], chunk_size=chunk_size, overlap=overlap):
                            # chunk도 sanitize되어야 함(_chunk_text에서 처리됨)
                            chunk_hash = _sha256_text(f"{doc_hash}:{page_no}:{chunk}")
                            chunk_rows.append({
                                "doc_title": doc_title,
                                "doc_group": doc_group,
                                "section": None,
                                "page": page_no,
                                "content": chunk,
                                "doc_path": str(pdf_path),
                                "doc_hash": doc_hash,
                                "chunk_hash": chunk_hash,
                            })

                    if not chunk_rows:
                        continue

                    chunks_attempted += len(chunk_rows)

                    # 2) embedding 생성
                    texts = [r["content"] for r in chunk_rows]
                    vecs = embed_texts(texts)

                    # 3) insert
                    before = cur.rowcount
                    for r, v in zip(chunk_rows, vecs):
                        cur.execute(insert_sql, {
                            "id": str(uuid.uuid4()),
                            **r,
                            "embedding": v,  # python list -> %s::vector 캐스팅
                        })
                        # ON CONFLICT DO NOTHING 일 때 rowcount가 0일 수 있음
                        if cur.rowcount == 1:
                            chunks_inserted += 1

                    conn.commit()

                    # rowcount는 누적/드라이버 차이가 있어서 안전하게 count로 확인
                    docs_indexed += 1

                except Exception as e:
                    conn.rollback()
                    docs_failed += 1
                    errors.append({
                        "pdf": str(pdf_path),
                        "error": str(e),
                    })
                    # 다음 파일 계속 진행
                    continue

            # ✅ START: after count
            cur.execute("SELECT COUNT(*) AS cnt FROM rag_chunks;")
            count_after = cur.fetchone()["cnt"]
            # ✅ END

    # 가장 정확한 inserted는 "ingest 전후 COUNT"로 잡는 방식이지만,
    # 지금은 응답을 단순화(실무에서는 trace/log로 더 정확히)
    return {
        "docs_indexed": docs_indexed,
        "docs_failed": docs_failed,
        "chunks_attempted": chunks_attempted,
        "chunks_inserted": chunks_inserted,
        "doc_root": str(root),
        "pdf_count": len(pdfs),
        "errors": errors[:5],  # 최대 5개 에러만 반환
    }

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
