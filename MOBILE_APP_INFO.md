# Mobile App Branch

이 브랜치(`mobile-app`)에는 SMS-Filtering 프로젝트의 모바일 애플리케이션 버전이 포함되어 있습니다.

## 디렉토리 구조

```
SMS-Filtering/
├── data/                  # 기존 프로젝트: 데이터셋
├── preprocess/            # 기존 프로젝트: 전처리 스크립트
├── utils/                 # 기존 프로젝트: 유틸리티
├── main.py                # 기존 프로젝트: 학습 스크립트
├── plm.py                 # 기존 프로젝트: 모델 정의
└── mobile-app/            # 🆕 모바일 앱 (React Native)
    ├── app/               # 화면 및 라우팅
    ├── components/        # UI 컴포넌트
    ├── server/            # 백엔드 API
    └── README.md          # 모바일 앱 문서
```

## 프로젝트 개요

### 기존 프로젝트 (PyTorch)
- BERT, ELECTRA, RoBERTa, BigBird 모델 학습
- 한국어 SMS 스팸 필터링
- 모델 평가 및 성능 비교

### 모바일 앱 (React Native)
- 기존 모델을 활용한 실시간 SMS 분석
- 사용자 친화적인 모바일 인터페이스
- LLM 기반 분류 시스템
- 로컬 저장소 및 통계 대시보드

## 시작하기

### 기존 프로젝트 실행
```bash
# 루트 디렉토리에서
pip install -r requirements.txt
python main.py --train --data_dir result --model_type bert
```

### 모바일 앱 실행
```bash
cd mobile-app
pnpm install
pnpm dev
```

자세한 내용은 `mobile-app/README.md`를 참조하세요.

## 브랜치 정보

- **main**: 기존 PyTorch 기반 프로젝트
- **mobile-app**: React Native 모바일 앱 추가 버전

## 기여

모바일 앱 관련 이슈 및 PR은 `mobile-app` 브랜치에 제출해주세요.
