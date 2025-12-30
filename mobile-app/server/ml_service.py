"""
SMS Spam Classification Model Service
FastAPI 기반 PyTorch 모델 추론 서버
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SMS Spam Classifier")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 타입 정의
ModelType = Literal["BERT", "RoBERTa", "BigBird"]

# 모델 매핑
MODEL_MAPPING = {
    "BERT": "monologg/kobert",
    "RoBERTa": "klue/roberta-base",
    "BigBird": "monologg/kobigbird-bert-base",
}

# 전역 모델 캐시
models_cache = {}
tokenizers_cache = {}


class AnalyzeRequest(BaseModel):
    text: str
    model: ModelType = "BERT"


class AnalyzeResponse(BaseModel):
    classification: Literal["SPAM", "INBOX"]
    confidence: float
    model: str
    reasoning: str


def load_model(model_type: ModelType):
    """모델과 토크나이저 로드 (캐싱)"""
    if model_type in models_cache:
        return models_cache[model_type], tokenizers_cache[model_type]
    
    try:
        model_name = MODEL_MAPPING[model_type]
        logger.info(f"Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        
        # 평가 모드로 설정
        model.eval()
        
        # 캐시에 저장
        models_cache[model_type] = model
        tokenizers_cache[model_type] = tokenizer
        
        logger.info(f"Model {model_type} loaded successfully")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model {model_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


def classify_text(text: str, model, tokenizer) -> tuple[str, float, str]:
    """텍스트 분류 수행"""
    try:
        # 토크나이징
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        )
        
        # 추론
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        # 결과 해석 (0: INBOX, 1: SPAM)
        spam_prob = probs[0][1].item()
        inbox_prob = probs[0][0].item()
        
        classification = "SPAM" if spam_prob > inbox_prob else "INBOX"
        confidence = max(spam_prob, inbox_prob)
        
        # 추론 근거 생성
        if classification == "SPAM":
            if spam_prob > 0.9:
                reasoning = "매우 높은 확률로 스팸 메시지로 판단됩니다."
            elif spam_prob > 0.7:
                reasoning = "스팸 메시지의 특징이 강하게 나타납니다."
            else:
                reasoning = "스팸 메시지일 가능성이 있습니다."
        else:
            if inbox_prob > 0.9:
                reasoning = "정상 메시지로 판단됩니다."
            elif inbox_prob > 0.7:
                reasoning = "정상 메시지의 특징을 보입니다."
            else:
                reasoning = "정상 메시지일 가능성이 높습니다."
        
        return classification, confidence, reasoning
    
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "status": "ok",
        "service": "SMS Spam Classifier",
        "models": list(MODEL_MAPPING.keys())
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_sms(request: AnalyzeRequest):
    """SMS 스팸 분류 API"""
    try:
        logger.info(f"Analyzing text with model: {request.model}")
        
        # 입력 검증
        if not request.text or len(request.text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > 1000:
            raise HTTPException(status_code=400, detail="Text too long (max 1000 characters)")
        
        # 모델 로드
        model, tokenizer = load_model(request.model)
        
        # 분류 수행
        classification, confidence, reasoning = classify_text(
            request.text, model, tokenizer
        )
        
        return AnalyzeResponse(
            classification=classification,
            confidence=confidence,
            model=request.model,
            reasoning=reasoning
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """사용 가능한 모델 목록"""
    return {
        "models": [
            {
                "name": model_type,
                "huggingface_id": model_name,
                "loaded": model_type in models_cache
            }
            for model_type, model_name in MODEL_MAPPING.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
