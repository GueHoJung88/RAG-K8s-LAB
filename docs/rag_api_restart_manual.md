# rag-api 서비스 재기동 매뉴얼

(이 문서는 ChatGPT 자동 생성 매뉴얼입니다)

## 1. 목적
- genai-k8s-lab 환경에서 rag-api 서비스가 접근되지 않거나 비정상 동작 시
- 원인을 계층별(Layer)로 분리하여 빠르게 복구하기 위함

## 2. 프로젝트 구조
```
genai-k8s-lab/
├─ k8s/
│  ├─ 10-postgres-pgvector.yaml
│  └─ 20-rag-api.yaml
├─ app/
│  └─ rag_api/
│     ├─ main.py
│     └─ rag.py
└─ /data/raw_docs
```

## 3. 재기동 표준 절차 요약
1) Pod 상태 확인
2) 로그 확인
3) Service / Endpoint 확인
4) port-forward 재연결
5) Health / Query 테스트

## 4. 가장 흔한 장애 원인
- port-forward 세션 종료
- Deployment 재기동 후 새 Pod 생성
- Ollama timeout
- HostPath 문서 미마운트

## 5. 핵심 명령어
```bash
kubectl -n genai rollout restart deploy/rag-api
kubectl -n genai port-forward svc/rag-api 8000:8000 --address 0.0.0.0
curl http://localhost:8000/health
```

## 6. 운영 팁
- ClusterIP 서비스는 외부 접근 불가 → port-forward 필수
- 장기 운영 시 Ingress 도입 권장
