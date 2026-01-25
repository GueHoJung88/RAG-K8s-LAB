from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from rag import retrieve


# Load environment variables from .env file
load_dotenv()

# -------------------------
# Settings (env-based)
# -------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ragpassword")

# Ollama endpoint (host machine)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Retrieval params
TOP_K = int(os.getenv("TOP_K", "4"))

app = FastAPI(title="GENAI RAG API", version="0.1.0")

# -------------------------
# Prometheus metrics
# -------------------------
REQ_COUNT = Counter("rag_api_requests_total", "Total requests", ["endpoint"])
REQ_LAT = Histogram("rag_api_request_latency_seconds", "Request latency", ["endpoint"])
RETRIEVAL_HITS = Counter("rag_retrieval_hits_total", "Retrieval hits", ["doc_group"])


# -------------------------
# Request/Response Schemas
# -------------------------
class QueryRequest(BaseModel):
    question: str
    doc_group: Optional[str] = None  # e.g. "law", "guide", "commentary", "ai_guideline"


class Evidence(BaseModel):
    doc_title: str
    doc_group: str
    section: Optional[str] = None
    page: Optional[int] = None
    score: float
    snippet: str


class QueryResponse(BaseModel):
    query_log_id: str
    answer: str
    evidences: List[Evidence]


class IngestRequest(BaseModel):
    # 이번 단계에서는 실제 PDF ingest 파이프라인을 아직 넣지 않고 "연결 테스트"를 위한 스텁을 제공합니다.
    # 다음 단계에서: PDF 파싱 → chunk → embedding → insert 로 확장합니다.
    ping: Optional[bool] = True


class FeedbackRequest(BaseModel):
    query_log_id: str
    rating: Optional[int] = None
    is_helpful: Optional[bool] = None
    is_grounded: Optional[bool] = None
    comment: Optional[str] = None



# -------------------------
# DB helpers
# -------------------------
def get_conn():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        cursor_factory=RealDictCursor,
    )


def ensure_vector_extension():
    # vector extension은 이미 Step 3에서 켰겠지만, 안전을 위해 한번 더 보장합니다.
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()


def fetch_topk_chunks(question: str, doc_group: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    MVP 단계에서는 'embedding 기반 벡터 검색'을 구현하기 전,
    DB에 미리 들어가 있는 chunk가 있다고 가정하고 "최근 chunk"를 top-k로 뽑는 스텁 형태입니다.

    다음 단계(Step 7~)에서:
      - sentence-transformers로 query embedding 생성
      - pgvector <-> 연산으로 top-k 유사도 검색
    로 교체합니다.
    """
    sql = """
        SELECT
          doc_title, doc_group, section, page,
          content,
          0.5::float as score
        FROM rag_chunks
        WHERE (%s IS NULL OR doc_group = %s)
        ORDER BY created_at DESC
        LIMIT %s
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (doc_group, doc_group, TOP_K))
            rows = cur.fetchall()
            return rows


def insert_query_log(
    query_log_id: str,
    question: str,
    doc_group: Optional[str],
    top_k: int,
    retrieved_meta: List[Dict[str, Any]],
    model_name: str,
    prompt_version: str,
    answer: str,
    t_retrieval_ms: int,
    t_generate_ms: int,
    t_total_ms: int,
):
    # score 요약
    scores = [float(x.get("score", 0.0)) for x in retrieved_meta if x.get("score") is not None]
    score_avg = sum(scores) / len(scores) if scores else 0.0
    score_min = min(scores) if scores else 0.0

    # citation coverage (간단 버전): 답변에 [숫자] 패턴 몇 개 포함?
    import re
    citation_coverage = len(re.findall(r"\[\d+\]", answer or ""))

    sql = """
    INSERT INTO rag_query_log (
      id, trace_id, user_id,
      question, doc_group, top_k,
      retrieved_count, retrieved_meta,
      model_name, prompt_version,
      answer, error,
      t_retrieval_ms, t_generate_ms, t_total_ms,
      citation_coverage, retrieval_score_avg, retrieval_score_min
    ) VALUES (
      %s, %s, %s,
      %s, %s, %s,
      %s, %s::jsonb,
      %s, %s,
      %s, NULL,
      %s, %s, %s,
      %s, %s, %s
    );
    """

    # trace/user는 오늘은 MVP라 None 처리(다음 단계에서 OIDC로 채우면 됨)
    trace_id = None
    user_id = None

    import json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    query_log_id, trace_id, user_id,
                    question, doc_group, top_k,
                    len(retrieved_meta), json.dumps(retrieved_meta),
                    model_name, prompt_version,
                    answer,
                    t_retrieval_ms, t_generate_ms, t_total_ms,
                    citation_coverage, score_avg, score_min,
                ),
            )
            conn.commit()



# -------------------------
# Ollama call
# -------------------------
def call_ollama(question: str, evidences: List[Dict[str, Any]]) -> str:
    # 근거 기반 답변을 강제하는 프롬프트
    evidence_texts = []
    for i, ev in enumerate(evidences, start=1):
        meta = f"[{i}] {ev.get('doc_title')} | group={ev.get('doc_group')} | section={ev.get('section')} | page={ev.get('page')}"
        snippet = (ev.get("content") or "")[:800]
        evidence_texts.append(f"{meta}\n{snippet}")

    prompt = f"""너는 개인정보보호 관련 문서를 근거로 답변하는 어시스턴트다.
아래 '근거'에 포함된 내용만 사용해 답변하라.
근거가 부족하면 '추가 확인 필요'로 명확히 말하고 추측하지 마라.

[질문]
{question}

[근거]
{chr(10).join(evidence_texts)}

[출력 형식]
1) 요약 답변
2) 법적/가이드 근거(근거 번호로 인용)
3) 실무 유의사항
4) 추가 확인 필요
"""

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama call failed: {e}")


# -------------------------
# Endpoints
# -------------------------
@app.on_event("startup")
def startup():
    ensure_vector_extension()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    """
    MVP 1단계: DB 연결/테이블 존재 여부를 확인하는 '연결 테스트' 성격.
    다음 단계에서 실제 PDF ingest 파이프라인으로 확장합니다.
    """
    start = time.time()
    REQ_COUNT.labels(endpoint="/ingest").inc()

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                _ = cur.fetchone()
        return {"status": "ok", "message": "DB connection OK. (Ingest pipeline will be added next.)"}
    finally:
        REQ_LAT.labels(endpoint="/ingest").observe(time.time() - start)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t0 = time.time()
    REQ_COUNT.labels(endpoint="/query").inc()

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    # 1) Retrieve (현재는 스텁: 최신 chunk top-k)
    # chunks = fetch_topk_chunks(req.question, req.doc_group)

    query_log_id = str(uuid.uuid4())

    # 1) Retrieve
    t_retrieval0 = time.time()
    # 실제 embedding 기반 검색으로 교체
    chunks = retrieve(req.question, req.doc_group) # rag.py :contentReference[oaicite:5]{index=5}
    t_retrieval_ms = int((time.time() - t_retrieval0) * 1000)

    if not chunks:
        # 로그도 남기고 싶다면 여기서 insert_query_log(error=...)로 확장 가능
        raise HTTPException(status_code=404, detail="No chunks found in DB. Ingest documents first.")

    for c in chunks:
        RETRIEVAL_HITS.labels(doc_group=c.get("doc_group", "unknown")).inc()

    # 2) Generate
    t_gen0 = time.time()
    answer = call_ollama(req.question, chunks)
    t_generate_ms = int((time.time() - t_gen0) * 1000)

    # 3) Evidence formatting
    evidences: List[Evidence] = []
    retrieved_meta: List[Dict[str, Any]] = []
    for idx, c in chunks:
        ev = Evidence(
                doc_title=c.get("doc_title", "unknown"),
                doc_group=c.get("doc_group", "unknown"),
                section=c.get("section"),
                page=c.get("page"),
                score=float(c.get("score", 0.0)),
                snippet=(c.get("content") or "")[:200],
            )
        evidences.append(ev)

        # DB에 저장할 meta (jsonb)
        retrieved_meta.append(
            {
                "rank": idx,
                "doc_title": ev.doc_title,
                "doc_group": ev.doc_group,
                "section": ev.section,
                "page": ev.page,
                "score": ev.score,
                "snippet": ev.snippet,
            }
        )

    t_total_ms = int((time.time() - t0) * 1000)

    # 4) LLMOps 로그 저장
    PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v0")
    insert_query_log(
        query_log_id=query_log_id,
        question=req.question,
        doc_group=req.doc_group,
        top_k=TOP_K,
        retrieved_meta=retrieved_meta,
        model_name=OLLAMA_MODEL,
        prompt_version=PROMPT_VERSION,
        answer=answer,
        t_retrieval_ms=t_retrieval_ms,
        t_generate_ms=t_generate_ms,
        t_total_ms=t_total_ms,
    )

    REQ_LAT.labels(endpoint="/query").observe(time.time() - t0)

    return QueryResponse(query_log_id=query_log_id, answer=answer, evidences=evidences)



@app.get("/metrics")
def metrics():
    start = time.time()
    REQ_COUNT.labels(endpoint="/metrics").inc()
    try:
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)
    finally:
        REQ_LAT.labels(endpoint="/metrics").observe(time.time() - start)



@app.post("/feedback")
def feedback(req: FeedbackRequest):
    fb_id = str(uuid.uuid4())
    sql = """
    INSERT INTO rag_feedback (id, query_log_id, rating, is_helpful, is_grounded, comment)
    VALUES (%s, %s, %s, %s, %s, %s);
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (fb_id, req.query_log_id, req.rating, req.is_helpful, req.is_grounded, req.comment),
            )
            conn.commit()
    return {"status": "ok", "feedback_id": fb_id}

