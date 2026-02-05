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

from rag import retrieve, ingest_pdfs_to_db


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

# -------------------------
# Ollama tuning knobs (env-based)
# -------------------------
# START: ollama_tuning_settings
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))   # read timeout
OLLAMA_CONNECT_TIMEOUT_SEC = int(os.getenv("OLLAMA_CONNECT_TIMEOUT_SEC", "5"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))   # max tokens to generate
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))
OLLAMA_REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.1"))
EVIDENCE_SNIPPET_CHARS = int(os.getenv("EVIDENCE_SNIPPET_CHARS", "450"))
# END: ollama_tuning_settings

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
    doc_group: str = "guide"
    limit: Optional[int] = None
    chunk_size: int = 1000
    overlap: int = 150


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


def insert_query_log(
    query_log_id: str,
    question: str,
    doc_group: Optional[str],
    top_k: int,
    retrieved_meta: List[Dict[str, Any]],
    embed_model_name: str,
    llm_model: str,
    prompt_version: str,
    answer: str,
    latency_ms: int,
    error: Optional[str] = None,
):
    """
    rag_query_log 스키마(우리가 생성한 버전)에 맞춰 INSERT.
    - evidences: jsonb 로 저장 (retrieved_meta 사용)
    - retrieval_*: 점수 요약
    - citation_coverage: [1][2] 같은 인용 번호 포함 수 (간단 버전)
    - context_overlap: 지금은 MVP로 0.0 (다음 단계에서 계산)
    """
    scores = [float(x.get("score", 0.0)) for x in retrieved_meta if x.get("score") is not None]
    score_avg = (sum(scores) / len(scores)) if scores else None
    score_min = min(scores) if scores else None
    # p95는 MVP에서는 생략/None 처리 (필요하면 numpy로 계산)
    score_p95 = None

    import re
    citation_coverage = len(re.findall(r"\[\d+\]", answer or ""))

    empty_retrieval = (len(retrieved_meta) == 0)

    # trace/user는 MVP라 None (나중에 OIDC로 채움)
    trace_id = None
    user_id = None
    user_role = None

    sql = """
    INSERT INTO rag_query_log (
      id,
      trace_id, user_id, user_role,
      question, doc_group,
      top_k,
      embed_model_name, llm_model, prompt_version,
      retrieval_count, retrieval_score_avg, retrieval_score_min, retrieval_score_p95,
      empty_retrieval,
      answer,
      evidences,
      latency_ms,
      citation_coverage,
      context_overlap,
      error
    )
    VALUES (
      %s,
      %s, %s, %s,
      %s, %s,
      %s,
      %s, %s, %s,
      %s, %s, %s, %s,
      %s,
      %s,
      %s::jsonb,
      %s,
      %s,
      %s,
      %s
    );
    """

    import json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    query_log_id,
                    trace_id, user_id, user_role,
                    question, doc_group,
                    top_k,
                    embed_model_name, llm_model, prompt_version,
                    len(retrieved_meta), score_avg, score_min, score_p95,
                    empty_retrieval,
                    answer,
                    json.dumps(retrieved_meta),
                    latency_ms,
                    float(citation_coverage),
                    0.0,  # context_overlap: MVP에서는 0.0 (다음 단계에서 계산)
                    error,
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
        # START: evidence_truncate
        snippet = (ev.get("content") or "")[:EVIDENCE_SNIPPET_CHARS]
        # END: evidence_truncate
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
        # START: ollama_options
        "options": {
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": OLLAMA_TEMPERATURE,
            "top_p": OLLAMA_TOP_P,
            "repeat_penalty": OLLAMA_REPEAT_PENALTY,
        },
        # END: ollama_options
    }

    try:
        r = requests.post(
            url,
            json=payload,
            timeout=(OLLAMA_CONNECT_TIMEOUT_SEC, OLLAMA_TIMEOUT_SEC),
        )  # START: ollama_timeout_tuple
        # END: ollama_timeout_tuple
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
        result = ingest_pdfs_to_db(
            doc_group=req.doc_group,
            limit=req.limit,
            chunk_size=req.chunk_size,
            overlap=req.overlap,
        )
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingest failed: {e}")
    finally:
        REQ_LAT.labels(endpoint="/ingest").observe(time.time() - start)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t0 = time.time()
    REQ_COUNT.labels(endpoint="/query").inc()

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")

    query_log_id = str(uuid.uuid4())

    # 1) Retrieve
    t_retrieval0 = time.time()
    chunks = retrieve(req.question, req.doc_group)
    t_retrieval_ms = int((time.time() - t_retrieval0) * 1000)

    if not chunks:
        # 검색 결과가 없더라도, 로그는 남기는 것이 실무적으로 유리
        insert_query_log(
            query_log_id=query_log_id,
            question=req.question,
            doc_group=req.doc_group,
            top_k=TOP_K,
            retrieved_meta=[],
            embed_model_name=os.getenv("EMBED_MODEL_NAME", "unknown"),
            llm_model=OLLAMA_MODEL,
            prompt_version=os.getenv("PROMPT_VERSION", "v0"),
            answer="",
            latency_ms=int((time.time() - t0) * 1000),
            error="empty retrieval",
        )
        raise HTTPException(status_code=404, detail="No chunks found in DB. Ingest documents first.")

    for c in chunks:
        RETRIEVAL_HITS.labels(doc_group=c.get("doc_group", "unknown")).inc()

    # 2) Generate
    t_gen0 = time.time()
    answer = call_ollama(req.question, chunks)
    t_generate_ms = int((time.time() - t_gen0) * 1000)

    # 3) Evidence formatting + retrieved_meta 구성
    evidences: List[Evidence] = []
    retrieved_meta: List[Dict[str, Any]] = []

    # ✅ 버그 수정: enumerate 사용
    for idx, c in enumerate(chunks, start=1):
        ev = Evidence(
            doc_title=c.get("doc_title", "unknown"),
            doc_group=c.get("doc_group", "unknown"),
            section=c.get("section"),
            page=c.get("page"),
            score=float(c.get("score", 0.0)),
            snippet=(c.get("content") or "")[:200],
        )
        evidences.append(ev)

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

    # 4) LLMOps 로그 저장 (스키마 정합성 OK)
    insert_query_log(
        query_log_id=query_log_id,
        question=req.question,
        doc_group=req.doc_group,
        top_k=TOP_K,
        retrieved_meta=retrieved_meta,
        embed_model_name=os.getenv("EMBED_MODEL_NAME", "unknown"),
        llm_model=OLLAMA_MODEL,
        prompt_version=os.getenv("PROMPT_VERSION", "v0"),
        answer=answer,
        latency_ms=int((time.time() - t0) * 1000),
        error=None,
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
