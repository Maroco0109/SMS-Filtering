# ML Service Integration

이 문서는 PyTorch 모델을 사용한 SMS 스팸 필터링 서비스 통합에 대해 설명합니다.

## 아키텍처

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│  Mobile App     │─────▶│  Node.js Server  │─────▶│  Python ML Service  │
│  (React Native) │      │  (Express/tRPC)  │      │  (FastAPI/PyTorch)  │
└─────────────────┘      └──────────────────┘      └─────────────────────┘
                                                              │
                                                              ▼
                                                    ┌─────────────────────┐
                                                    │  Hugging Face       │
                                                    │  Transformers       │
                                                    │  - BERT             │
                                                    │  - RoBERTa          │
                                                    │  - BigBird          │
                                                    └─────────────────────┘
```

## 구성 요소

### 1. Python ML Service (`ml_service.py`)

FastAPI 기반의 PyTorch 모델 추론 서버입니다.

**기능**:
- BERT, RoBERTa, BigBird 모델 로딩 및 캐싱
- SMS 텍스트 분류 (SPAM/INBOX)
- 신뢰도 점수 및 추론 근거 제공

**엔드포인트**:
- `GET /`: 헬스 체크
- `POST /analyze`: SMS 분석
- `GET /models`: 사용 가능한 모델 목록

### 2. ML Client (`ml_client.ts`)

Node.js 서버에서 Python ML 서비스를 호출하는 클라이언트입니다.

**기능**:
- HTTP 요청을 통한 ML 서비스 통신
- 헬스 체크 및 재시도 로직
- 에러 핸들링 및 타임아웃 관리

### 3. tRPC Router (`routers.ts`)

모바일 앱과 통신하는 API 라우터입니다.

**기능**:
- PyTorch 모델 우선 사용
- ML 서비스 장애 시 LLM 폴백
- 유연한 모델 선택

## 설치 및 실행

### Python ML Service 설정

```bash
cd server

# 가상환경 생성
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r ml_requirements.txt

# 서버 시작
python ml_service.py
```

또는 스크립트 사용:

```bash
chmod +x start_ml_service.sh
./start_ml_service.sh
```

서버는 `http://localhost:8000`에서 실행됩니다.

### Node.js Server 설정

```bash
# 프로젝트 루트에서
pnpm install
pnpm dev
```

Node.js 서버는 자동으로 Python ML 서비스에 연결을 시도합니다.

## 환경 변수

`.env` 파일에 다음 변수를 추가할 수 있습니다:

```bash
# ML Service URL (기본값: http://localhost:8000)
ML_SERVICE_URL=http://localhost:8000
```

## 모델 정보

### 지원 모델

| 모델 | Hugging Face ID | 설명 |
|------|----------------|------|
| BERT | `monologg/kobert` | 한국어 BERT 모델 |
| RoBERTa | `klue/roberta-base` | KLUE RoBERTa 베이스 모델 |
| BigBird | `monologg/kobigbird-bert-base` | 한국어 BigBird 모델 |

### 모델 로딩

- 첫 요청 시 모델이 자동으로 다운로드됩니다 (수 GB)
- 이후 요청은 캐시된 모델을 사용합니다
- 모델은 메모리에 캐싱되어 빠른 추론을 제공합니다

## API 사용 예시

### Python ML Service 직접 호출

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "축하합니다! 1억원 당첨! 지금 바로 클릭하세요!",
    "model": "BERT"
  }'
```

**응답**:
```json
{
  "classification": "SPAM",
  "confidence": 0.95,
  "model": "BERT",
  "reasoning": "매우 높은 확률로 스팸 메시지로 판단됩니다."
}
```

### tRPC API 호출 (모바일 앱에서)

```typescript
const result = await trpc.sms.analyze.mutate({
  text: "축하합니다! 1억원 당첨!",
  model: "BERT",
  useLLM: false, // PyTorch 모델 사용
});

console.log(result);
// {
//   classification: "SPAM",
//   confidence: 0.95,
//   model: "BERT",
//   reasoning: "...",
//   source: "pytorch"
// }
```

## 폴백 메커니즘

시스템은 다음과 같은 폴백 전략을 사용합니다:

1. **PyTorch 모델 우선**: ML 서비스가 사용 가능하면 PyTorch 모델 사용
2. **LLM 폴백**: ML 서비스 장애 시 자동으로 LLM으로 전환
3. **명시적 LLM 사용**: `useLLM: true` 옵션으로 LLM 직접 사용 가능

```typescript
// PyTorch 모델 사용 (기본)
await trpc.sms.analyze.mutate({ text, model: "BERT" });

// LLM 강제 사용
await trpc.sms.analyze.mutate({ text, model: "BERT", useLLM: true });
```

## 성능 최적화

### 모델 캐싱

- 모델은 첫 로드 후 메모리에 캐싱됩니다
- 동일 모델에 대한 후속 요청은 즉시 처리됩니다

### 타임아웃 설정

- 모델 로딩: 최대 30초
- 추론: 일반적으로 1초 이내

### 리소스 요구사항

- **메모리**: 모델당 약 500MB - 1GB
- **CPU**: 추론 시 CPU 사용 (GPU 선택 사항)
- **디스크**: 모델 캐시용 약 2-3GB

## 트러블슈팅

### ML 서비스가 시작되지 않음

```bash
# Python 버전 확인 (3.8+ 필요)
python3 --version

# 의존성 재설치
pip install --upgrade -r ml_requirements.txt

# 포트 충돌 확인
lsof -i :8000
```

### 모델 다운로드 실패

```bash
# Hugging Face 캐시 디렉토리 확인
echo $HF_HOME  # 기본값: ~/.cache/huggingface

# 수동 다운로드
python -c "from transformers import AutoModel; AutoModel.from_pretrained('monologg/kobert')"
```

### 메모리 부족

- 한 번에 하나의 모델만 로드하도록 설정
- 경량 모델 사용 (BERT 대신 DistilBERT)
- 모델 양자화 적용

## 프로덕션 배포

### Docker 컨테이너화

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY ml_requirements.txt .
RUN pip install --no-cache-dir -r ml_requirements.txt

COPY ml_service.py .
EXPOSE 8000

CMD ["python", "ml_service.py"]
```

### 스케일링

- 여러 ML 서비스 인스턴스 실행
- 로드 밸런서 사용
- 모델 서빙 플랫폼 (TorchServe, TensorFlow Serving) 고려

## 참고 자료

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
