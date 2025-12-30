# SMS Shield - Mobile App

React Native 기반의 AI 스팸 메시지 필터링 모바일 애플리케이션입니다.

## 개요

이 디렉토리는 기존 SMS-Filtering 프로젝트의 모바일 앱 버전입니다. BERT, RoBERTa, BigBird 모델을 활용한 LLM 기반 스팸 분류 시스템을 모바일 환경에서 사용할 수 있도록 구현했습니다.

## 주요 기능

- **AI 기반 SMS 분석**: BERT, RoBERTa, BigBird 모델 선택 가능
- **실시간 분류**: 입력된 메시지를 즉시 SPAM/INBOX로 분류
- **메시지 관리**: 분석된 메시지 저장, 필터링, 삭제
- **통계 대시보드**: 오늘 차단된 스팸 수, 전체 메시지 통계
- **설정 관리**: 기본 모델 선택, 알림 설정, 데이터 관리

## 기술 스택

- **Frontend**: React Native, Expo SDK 54, TypeScript
- **Styling**: NativeWind 4 (Tailwind CSS)
- **Backend**: Express, tRPC
- **AI**: LLM Integration (Manus Platform)
- **Storage**: AsyncStorage (로컬)
- **State Management**: React Context API

## 프로젝트 구조

```
mobile-app/
├── app/                    # 화면 및 라우팅
│   ├── (tabs)/            # 탭 네비게이션
│   │   ├── index.tsx      # Home 화면
│   │   ├── messages.tsx   # Messages 화면
│   │   └── settings.tsx   # Settings 화면
│   └── analyze.tsx        # 메시지 분석 화면
├── components/            # 재사용 컴포넌트
├── lib/                   # 유틸리티 및 프로바이더
│   ├── message-provider.tsx
│   ├── settings-provider.tsx
│   └── storage.ts
├── server/                # 백엔드 API
│   └── routers.ts         # tRPC 라우터
├── types/                 # TypeScript 타입
└── tests/                 # 테스트 파일
```

## 설치 및 실행

### 사전 요구사항
- Node.js 22+
- pnpm 9+

### 설치
```bash
cd mobile-app
pnpm install
```

### 개발 서버 실행
```bash
pnpm dev
```

### 모바일에서 테스트
1. Expo Go 앱 설치
2. QR 코드 스캔
3. 앱 실행

## API 엔드포인트

### POST /api/sms/analyze
```typescript
// 요청
{
  text: string;
  model: "BERT" | "RoBERTa" | "BigBird";
}

// 응답
{
  classification: "SPAM" | "INBOX";
  confidence: number;
  model: string;
  reasoning: string;
}
```

## 기존 프로젝트와의 관계

이 모바일 앱은 부모 디렉토리의 SMS-Filtering 프로젝트를 기반으로 합니다:

- **기존 프로젝트**: PyTorch 기반 모델 학습 및 평가
- **모바일 앱**: LLM을 활용한 실시간 분류 및 사용자 인터페이스

## 향후 계획

- [ ] 실제 SMS 권한 통합 (Android)
- [ ] 데이터베이스 연동 (클라우드 동기화)
- [ ] 푸시 알림 기능
- [ ] 통계 차트 및 분석 리포트
- [ ] 실제 BERT/RoBERTa/BigBird 모델 배포

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.
