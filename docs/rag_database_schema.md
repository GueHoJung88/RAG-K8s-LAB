# RAG 데이터베이스 스키마 (PostgreSQL + pgvector)

## 개요

이 문서는 GENAI RAG 실무형 데이터베이스 스키마를 설명합니다. PostgreSQL + pgvector를 사용하여 문서 청크, 쿼리 로그, 사용자 피드백을 저장하고 관리합니다.

### 스키마 버전
- **버전**: v1
- **적용 날짜**: 2026-01-28
- **적용 방법**: kubectl exec를 통한 직접 실행

### 주요 특징
- **벡터 검색**: pgvector를 활용한 시맨틱 검색
- **감사 로그**: 쿼리 로그 및 사용자 피드백 저장
- **중복 방지**: 해시 기반 청크 중복 방지
- **성능 최적화**: 적절한 인덱스 구성

---

## 1. 사전 준비

### pgvector 확장 설치
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 적용 명령어
```bash
kubectl -n genai exec -it deploy/postgres-pgvector -- bash -lc \
'psql -U raguser -d ragdb -v ON_ERROR_STOP=1 << "SQL"
[SQL 스크립트 내용]
SQL'
```

---

## 2. 테이블 구조

### 2.1 rag_chunks (문서 청크 저장)

문서 청크와 벡터 임베딩을 저장하는 메인 테이블입니다.

```sql
CREATE TABLE rag_chunks (
  id          uuid PRIMARY KEY,
  doc_title   text NOT NULL,
  doc_group   text NOT NULL,             -- 예: guide, law, ai_guideline 등
  section     text,
  page        int,
  content     text NOT NULL,

  embedding   vector(384),               -- all-MiniLM-L6-v2 = 384 dims

  created_at  timestamptz NOT NULL DEFAULT now(),

  -- 증분/중복 방지용 메타
  doc_path    text,                      -- 파일 경로
  doc_hash    text,                      -- 문서 전체 텍스트 해시
  chunk_hash  text                       -- (doc_hash + page + chunk_text) 해시
);
```

#### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `id` | uuid | 기본키 |
| `doc_title` | text | 문서 제목 |
| `doc_group` | text | 문서 그룹 (guide, law, ai_guideline 등) |
| `section` | text | 섹션/챕터 정보 |
| `page` | int | 페이지 번호 |
| `content` | text | 청크 텍스트 내용 |
| `embedding` | vector(384) | 벡터 임베딩 (384차원) |
| `created_at` | timestamptz | 생성 시각 |
| `doc_path` | text | 원본 파일 경로 |
| `doc_hash` | text | 문서 전체 해시 (중복 방지) |
| `chunk_hash` | text | 청크 해시 (중복 방지) |

#### 인덱스
```sql
-- 중복 방지
CREATE UNIQUE INDEX uq_rag_chunks_chunk_hash ON rag_chunks (chunk_hash);

-- 필터링 + 최신순 정렬
CREATE INDEX idx_rag_chunks_group_created ON rag_chunks (doc_group, created_at DESC);

-- 벡터 검색 (IVFFlat)
CREATE INDEX idx_rag_chunks_embedding_ivfflat
ON rag_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

---

### 2.2 rag_query_log (쿼리 로그)

사용자 쿼리, 검색 결과, 응답을 로깅하는 테이블입니다.

```sql
CREATE TABLE rag_query_log (
  id               uuid PRIMARY KEY,
  created_at       timestamptz NOT NULL DEFAULT now(),

  -- 권한/감사/추적
  trace_id         text,                 -- 앱/게이트웨이 trace id
  user_id          text,                 -- SSO 연동 시 sub/user id
  user_role        text,                 -- admin, auditor, user 등

  -- 입력
  question         text NOT NULL,
  doc_group        text,

  -- 검색/생성 파라미터
  top_k            int,
  embed_model_name text,
  llm_model        text,
  prompt_version   text,

  -- 검색 결과 요약 지표
  retrieval_count  int,
  retrieval_score_avg float,
  retrieval_score_min float,
  retrieval_score_p95 float,
  empty_retrieval  boolean DEFAULT false,

  -- 응답
  answer           text,

  -- 근거 (JSONB)
  evidences        jsonb,

  -- 성능/품질 지표
  latency_ms       int,
  citation_coverage float,               -- 근거 인용 커버리지
  context_overlap  float,                -- groundedness 지표

  error            text                  -- 실패/예외 메시지
);
```

#### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `id` | uuid | 기본키 |
| `created_at` | timestamptz | 쿼리 시각 |
| `trace_id` | text | 트레이스 ID (분산 추적) |
| `user_id` | text | 사용자 ID |
| `user_role` | text | 사용자 역할 |
| `question` | text | 사용자 질문 |
| `doc_group` | text | 검색 대상 그룹 |
| `top_k` | int | 검색할 청크 수 |
| `embed_model_name` | text | 임베딩 모델명 |
| `llm_model` | text | LLM 모델명 |
| `prompt_version` | text | 프롬프트 버전 |
| `retrieval_count` | int | 검색된 청크 수 |
| `retrieval_score_avg` | float | 평균 유사도 점수 |
| `retrieval_score_min` | float | 최소 유사도 점수 |
| `retrieval_score_p95` | float | 95% 백분위 점수 |
| `empty_retrieval` | boolean | 검색 결과 없음 여부 |
| `answer` | text | 생성된 답변 |
| `evidences` | jsonb | 근거 청크 정보 (JSON) |
| `latency_ms` | int | 응답 시간 (ms) |
| `citation_coverage` | float | 인용 커버리지 |
| `context_overlap` | float | 컨텍스트 일치도 |
| `error` | text | 에러 메시지 |

#### evidences JSON 구조 예시:
```json
[
  {
    "chunk_id": "uuid-here",
    "doc_title": "문서 제목",
    "page": 12,
    "score": 0.71,
    "snippet": "청크 내용 일부..."
  }
]
```

#### 인덱스
```sql
CREATE INDEX idx_rag_query_log_created ON rag_query_log (created_at DESC);
CREATE INDEX idx_rag_query_log_user_created ON rag_query_log (user_id, created_at DESC);
CREATE INDEX idx_rag_query_log_trace ON rag_query_log (trace_id);
```

---

### 2.3 rag_feedback (사용자 피드백)

사용자 피드백을 저장하는 테이블입니다.

```sql
CREATE TABLE rag_feedback (
  id            uuid PRIMARY KEY,
  created_at    timestamptz NOT NULL DEFAULT now(),

  -- FK
  query_log_id  uuid NOT NULL REFERENCES rag_query_log(id) ON DELETE CASCADE,

  -- 피드백 주체
  user_id       text,
  rating        int,                      -- 1~5 또는 -1/1
  is_helpful    boolean,                  -- thumb up/down
  feedback_text text,                     -- 자유 코멘트
  tags          text[],                   -- 태그 배열

  -- 운영/품질 개선용
  expected_answer text,
  chosen_evidence_ids uuid[]              -- 선택된 근거 ID들
);
```

#### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `id` | uuid | 기본키 |
| `created_at` | timestamptz | 피드백 시각 |
| `query_log_id` | uuid | 참조하는 쿼리 로그 ID (FK) |
| `user_id` | text | 피드백 제공자 |
| `rating` | int | 평점 (1-5 또는 -1/1) |
| `is_helpful` | boolean | 도움이 되었는지 |
| `feedback_text` | text | 자유 텍스트 피드백 |
| `tags` | text[] | 태그 배열 |
| `expected_answer` | text | 기대했던 답변 |
| `chosen_evidence_ids` | uuid[] | 선택된 근거 청크 ID들 |

#### 인덱스
```sql
CREATE INDEX idx_rag_feedback_created ON rag_feedback (created_at DESC);
CREATE INDEX idx_rag_feedback_query ON rag_feedback (query_log_id, created_at DESC);
```

---

## 3. 성능 최적화

### ANALYZE 실행
데이터 적재 후 통계 정보 업데이트:
```sql
ANALYZE rag_chunks;
ANALYZE rag_query_log;
ANALYZE rag_feedback;
```

### 벡터 검색 성능
- IVFFlat 인덱스는 데이터 양에 따라 `lists` 파라미터 조정 필요
- 프로브 수(`lists`)가 많을수록 정확도 ↑, 속도 ↓
- 실무에서는 100-1000 사이 값 사용

---

## 4. 사용 예시

### 청크 삽입
```sql
INSERT INTO rag_chunks (
  id, doc_title, doc_group, content, embedding,
  doc_path, doc_hash, chunk_hash
) VALUES (
  gen_random_uuid(),
  'AI 개발 가이드',
  'guide',
  '청크 내용...',
  '[0.1, 0.2, ...]'::vector,
  '/path/to/doc.pdf',
  'hash123',
  'chunk_hash456'
);
```

### 벡터 검색
```sql
SELECT id, doc_title, content, 1 - (embedding <=> '[쿼리 벡터]'::vector) as score
FROM rag_chunks
WHERE doc_group = 'guide'
ORDER BY embedding <=> '[쿼리 벡터]'::vector
LIMIT 5;
```

### 쿼리 로그 기록
```sql
INSERT INTO rag_query_log (
  id, question, answer, evidences, latency_ms
) VALUES (
  gen_random_uuid(),
  '질문 내용',
  '답변 내용',
  '[{"chunk_id": "...", "score": 0.8}]'::jsonb,
  1500
);
```

---

## 5. 모니터링 및 유지보수

### 테이블 크기 확인
```sql
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
  AND tablename LIKE 'rag_%';
```

### 인덱스 사용률 확인
```sql
SELECT schemaname, tablename, indexname,
       idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
  AND tablename LIKE 'rag_%';
```

### VACUUM 및 REINDEX
```sql
VACUUM ANALYZE rag_chunks;
REINDEX INDEX idx_rag_chunks_embedding_ivfflat;
```

---

## 6. 확장 고려사항

### 데이터 파티셔닝
- `rag_query_log`: 날짜별 파티셔닝 고려
- 대용량 데이터 시 성능 개선

### 백업 전략
- 벡터 데이터 포함 전체 백업
- 증분 백업 고려

### 모니터링
- 쿼리 성능 모니터링
- 벡터 검색 정확도 추적
- 사용자 피드백 기반 품질 개선</content>
<parameter name="filePath">/home/addinedu/Documents/AI_FT_RAG/dev/genai-k8s-lab/docs/rag_database_schema.md