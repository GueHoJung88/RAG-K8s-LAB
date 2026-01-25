import os
import re
import uuid
from dotenv import load_dotenv
from pypdf import PdfReader
import psycopg2
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
# 프로젝트 루트 기준으로 경로 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(project_root, "tools/.env"))

# K8s Postgres로 port-forward 붙을 것이므로 localhost 사용
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "ragdb")
DB_USER = os.getenv("DB_USER", "raguser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ragpassword")

DOC_ROOT = os.getenv("DOC_ROOT", os.path.join(project_root, "data/raw_docs"))
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# 384-dim, CPU에서도 빠름
model = SentenceTransformer(EMBED_MODEL_NAME)

def get_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )

def extract_pages(pdf_path: str):
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # NUL 문자 제거 (PostgreSQL이 지원하지 않음)
        text = text.replace("\x00", "")
        yield i, text

def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return
    i = 0
    while i < len(text):
        yield text[i:i+max_chars]
        i += max_chars - overlap

def infer_doc_group(path: str) -> str:
    p = path.replace("\\", "/").split("/")
    if "raw_docs" in p:
        idx = p.index("raw_docs")
        if idx + 1 < len(p):
            return p[idx + 1]
    return "unknown"

def main():
    pdf_files = []
    for root, _, files in os.walk(DOC_ROOT):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))

    print(f"PDF count: {len(pdf_files)}")
    if not pdf_files:
        raise SystemExit(f"No PDF found. Check DOC_ROOT={DOC_ROOT}")

    insert_sql = """
    INSERT INTO rag_chunks
      (id, content, embedding, doc_group, doc_title, doc_version, source_org, section, page, file_path)
    VALUES
      (%s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s)
    """

    inserted = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for pdf_path in pdf_files:
                doc_group = infer_doc_group(pdf_path)
                doc_title = os.path.splitext(os.path.basename(pdf_path))[0]

                for page_no, page_text in extract_pages(pdf_path):
                    for ch in chunk_text(page_text):
                        vec = model.encode([ch], normalize_embeddings=True)[0].tolist()
                        cur.execute(
                            insert_sql,
                            (
                                str(uuid.uuid4()),
                                ch,
                                vec,
                                doc_group,
                                doc_title,
                                None,
                                None,
                                None,
                                page_no,
                                pdf_path,
                            ),
                        )
                        inserted += 1

                conn.commit()
                print(f"[OK] {doc_title} | inserted so far: {inserted}")

    print(f"Done. Total inserted chunks: {inserted}")

if __name__ == "__main__":
    main()
