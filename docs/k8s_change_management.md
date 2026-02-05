# K8s 변경사항 적용 가이드

## 개요
K8s 환경의 변경사항은 **변경 범위와 성질**에 따라 다른 작업 절차가 필요합니다. 이 문서는 주요 변경 시나리오별로 필요한 작업 단계를 정리합니다.

---

## 1️⃣ 소스 코드 변경 (Source Code)

### 변경 대상
- Python 파일 (`.py`) 로직 수정
- 애플리케이션 코드 변경

### 적용 절차

```
1. 소스 코드 수정
   └─ app/rag_api/main.py, rag.py 등 수정

2. Docker 이미지 재빌드
   └─ docker build -t rag-api:0.5 app/
   └─ 버전 태그 변경 (0.4 → 0.5)

3. K8s 매니페스트 업데이트
   └─ k8s/20-rag-api.yaml의 image 필드 변경
   └─ image: rag-api:0.5

4. K8s 리소스 재배포
   └─ kubectl apply -f k8s/20-rag-api.yaml

5. 파드 재시작
   └─ kubectl rollout restart deployment/rag-api -n genai
   └─ 또는: kubectl delete pod -l app=rag-api -n genai
```

### 예시
```yaml
# 변경 전
image: rag-api:0.4

# 변경 후
image: rag-api:0.5
```

---

## 2️⃣ IP 주소/호스트명 변경 (네트워크 설정)

### 변경 대상
- 환경변수에 포함된 IP 주소
- 데이터베이스 호스트, Ollama URL 등
- 예: `OLLAMA_BASE_URL`, `DB_HOST` 등

### 적용 절차

```
1. YAML 파일 수정
   └─ k8s/20-rag-api.yaml의 env 섹션 수정
   └─ 새로운 IP/호스트명으로 변경

2. K8s 리소스 재배포
   └─ kubectl apply -f k8s/20-rag-api.yaml

3. 파드 재시작 (필수)
   └─ kubectl rollout restart deployment/rag-api -n genai
   └─ 환경변수 적용을 위해 파드 재시작 필요
```

### 예시
```yaml
# 변경 전
env:
  - name: OLLAMA_BASE_URL
    value: "http://192.168.0.69:11434"

# 변경 후
env:
  - name: OLLAMA_BASE_URL
    value: "http://192.168.0.46:11434"
```

### 적용 명령어
```bash
kubectl set env deployment/rag-api OLLAMA_BASE_URL=http://192.168.0.46:11434 -n genai
# 또는
kubectl apply -f k8s/20-rag-api.yaml
kubectl rollout restart deployment/rag-api -n genai
```

---

## 3️⃣ 환경변수/설정값 변경

### 변경 대상
- 애플리케이션 설정값
- 데이터베이스 자격증명
- 모델 이름, TOP_K, 배치 크기 등

### 적용 절차

```
1. YAML 파일의 env 섹션 수정
   └─ k8s/20-rag-api.yaml 수정

2. K8s 리소스 재배포
   └─ kubectl apply -f k8s/20-rag-api.yaml

3. 파드 재시작
   └─ kubectl rollout restart deployment/rag-api -n genai
```

### 예시
```yaml
# 변경 전
- name: TOP_K
  value: "4"
- name: OLLAMA_MODEL
  value: "llama3.2:3b"

# 변경 후
- name: TOP_K
  value: "8"
- name: OLLAMA_MODEL
  value: "llama2:7b"
```

---

## 4️⃣ 리소스 할당 변경 (CPU, Memory)

### 변경 대상
- `resources.requests` (최소 보장)
- `resources.limits` (최대 허용)

### 적용 절차

```
1. YAML 파일의 resources 섹션 수정
   └─ k8s/20-rag-api.yaml 수정

2. K8s 리소스 재배포
   └─ kubectl apply -f k8s/20-rag-api.yaml

3. 파드 재시작
   └─ kubectl rollout restart deployment/rag-api -n genai
   └─ 리소스 재할당이 필요한 경우 파드 재시작
```

### 예시
```yaml
# 변경 전
resources:
  requests:
    memory: "512Mi"
    cpu: "300m"
  limits:
    memory: "2Gi"
    cpu: "1"

# 변경 후
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "2"
```

---

## 5️⃣ 레플리카 수 변경 (스케일링)

### 변경 대상
- `spec.replicas` 값

### 적용 절차

```
1. YAML 파일의 replicas 변경
   └─ k8s/20-rag-api.yaml 수정

2. K8s 리소스 재배포
   └─ kubectl apply -f k8s/20-rag-api.yaml
   └─ 자동으로 파드가 추가/제거됨 (재시작 불필요)
```

### 예시
```yaml
# 변경 전
spec:
  replicas: 1

# 변경 후
spec:
  replicas: 3
```

### 빠른 적용 (YAML 수정 없이)
```bash
kubectl scale deployment/rag-api --replicas=3 -n genai
```

---

## 6️⃣ 새로운 리소스 추가

### 변경 대상
- 새로운 Deployment, Service, ConfigMap 등 추가
- 기존 리소스에 새로운 컨테이너 추가

### 적용 절차

```
1. 새 YAML 파일 생성 또는 기존 파일에 추가
   └─ --- 구분자로 분리

2. 첫 배포 시 kubectl apply 실행
   └─ kubectl apply -f k8s/새파일.yaml
   └─ 또는 kubectl apply -f k8s/

3. 이후 수정 사항은 동일하게 kubectl apply 적용
```

### 예시 (새 ConfigMap 추가)
```yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: genai
data:
  config.json: |
    {
      "temperature": 0.7,
      "max_tokens": 2048
    }
```

---

## 7️⃣ 볼륨/스토리지 설정 변경

### 변경 대상
- PersistentVolumeClaim (PVC) 용량
- 마운트 경로
- 스토리지 클래스

### 적용 절차

```
1. YAML 파일 수정
   └─ k8s/10-postgres-pgvector.yaml 수정

2. 용량 증가만 가능한 경우
   └─ kubectl apply -f k8s/10-postgres-pgvector.yaml
   └─ PVC가 증가된 용량으로 업데이트됨

3. 용량 감소 등 주요 변경
   └─ PVC 삭제 후 재생성 필요 (데이터 손실 주의)
   └─ kubectl delete pvc postgres-pvc -n genai
   └─ kubectl apply -f k8s/10-postgres-pgvector.yaml
```

### 예시
```yaml
# 변경 전
spec:
  resources:
    requests:
      storage: 10Gi

# 변경 후
spec:
  resources:
    requests:
      storage: 50Gi
```

---

## 8️⃣ 포트/서비스 설정 변경

### 변경 대상
- Service의 port 매핑
- containerPort 변경
- 서비스 타입 (ClusterIP → LoadBalancer 등)

### 적용 절차

```
1. YAML 파일의 ports 섹션 수정
   └─ k8s/20-rag-api.yaml 수정

2. K8s 리소스 재배포
   └─ kubectl apply -f k8s/20-rag-api.yaml

3. 포트 매핑 변경 시 파드 재시작
   └─ kubectl rollout restart deployment/rag-api -n genai
```

### 예시
```yaml
# 변경 전
ports:
  - containerPort: 8000

# 변경 후
ports:
  - containerPort: 8080
```

---

## 변경 적용 체크리스트

### 모든 변경 시 공통 작업
- [ ] YAML 파일 수정 및 저장
- [ ] `kubectl apply -f k8s/파일.yaml` 실행
- [ ] 변경 내용 확인: `kubectl get deployment -n genai -o wide`

### 환경변수/설정 변경 시
- [ ] 파드 재시작: `kubectl rollout restart deployment/rag-api -n genai`

### 소스 코드 변경 시
- [ ] Docker 이미지 재빌드
- [ ] 이미지 태그 버전 업데이트
- [ ] YAML 파일의 image 필드 업데이트
- [ ] kubectl apply 실행
- [ ] 파드 재시작

### 검증 명령어

```bash
# 배포 상태 확인
kubectl get deployment -n genai
kubectl get pods -n genai

# 파드의 환경변수 확인
kubectl exec -it <pod-name> -n genai -- env

# 로그 확인
kubectl logs <pod-name> -n genai

# 롤아웃 상태 확인
kubectl rollout status deployment/rag-api -n genai

# 서비스 확인
kubectl get svc -n genai
```

---

## 빠른 참조

| 변경 종류 | 파일 수정 | apply | 파드 재시작 | 이미지 재빌드 |
|---------|---------|-------|----------|------------|
| 소스 코드 | ✅ | ✅ | ✅ | ✅ |
| IP/호스트명 | ✅ | ✅ | ✅ | ❌ |
| 환경변수 | ✅ | ✅ | ✅ | ❌ |
| 리소스 할당 | ✅ | ✅ | ✅ | ❌ |
| 레플리카 수 | ✅ | ✅ | ❌ | ❌ |
| 스토리지 용량 | ✅ | ✅ | ❌ | ❌ |
| 포트 변경 | ✅ | ✅ | ✅ | ❌ |

---

## 팁

### 한 번에 여러 파일 적용
```bash
kubectl apply -f k8s/
```

### 드라이 런 (실제 적용 전 확인)
```bash
kubectl apply -f k8s/20-rag-api.yaml --dry-run=client
```

### 변경 사항 보기
```bash
kubectl diff -f k8s/20-rag-api.yaml
```

### 무중단 배포 (Rolling Update)
```bash
# 기본 설정으로 자동 처리됨
kubectl rollout restart deployment/rag-api -n genai

# 롤아웃 상태 모니터링
kubectl rollout status deployment/rag-api -n genai --watch
```

### 이전 버전으로 롤백
```bash
kubectl rollout undo deployment/rag-api -n genai
kubectl rollout history deployment/rag-api -n genai
```
