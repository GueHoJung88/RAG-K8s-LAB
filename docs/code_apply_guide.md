## 🚀 코드 수정 후 반영 프로세스 (Standard Flow)
가장 권장되는 방법은 이미지 태그(버전)를 변경하여 쿠버네티스가 "오, 새로운 버전이 나왔네?"라고 인식하게 만드는 것입니다.

### 1단계: 도커 이미지 빌드 (태그 변경)
기존 0.1에서 0.2로 버전을 올려서 빌드합니다.

```Bash
docker build -t rag-api:0.2 .
```

### 2단계: kind 클러스터로 이미지 로드
kind는 로컬 도커 엔진과 분리되어 있으므로, 새로 만든 이미지를 클러스터 내부로 넣어줘야 합니다.

```Bash
kind load docker-image rag-api:0.2
```

### 3단계: YAML 파일 수정 및 적용
k8s/20-rag-api.yaml 파일에서 이미지 주소를 rag-api:0.2로 변경합니다.

```YAML

spec:
  containers:
    - name: rag-api
      image: rag-api:0.2  # 0.1에서 0.2로 수정
```

그 다음 apply를 실행합니다.

```Bash
kubectl apply -f k8s/20-rag-api.yaml
```

결과: 쿠버네티스가 이미지가 변경된 것을 감지하고 자동으로 기존 Pod를 죽이고 새 Pod를 띄웁니다. (rollout restart를 별도로 할 필요가 없습니다.)