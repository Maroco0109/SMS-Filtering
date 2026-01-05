# SMS Shield - AI 기반 스팸 메시지 필터링 앱

SMS Shield는 AI 기술을 활용하여 스팸 메시지를 자동으로 필터링하는 모바일 애플리케이션입니다. BERT, RoBERTa, BigBird 모델을 기반으로 한 LLM 분류 시스템을 통해 높은 정확도의 스팸 탐지를 제공합니다.

## 주요 기능

### 1. AI 기반 메시지 분석
- **다중 모델 지원**: BERT, RoBERTa, BigBird 중 선택 가능
- **실시간 분석**: 입력된 SMS 메시지를 즉시 분석하여 SPAM 또는 INBOX로 분류
- **신뢰도 점수**: 각 분류 결과에 대한 신뢰도(0-100%) 제공

### 2. 메시지 관리
- **자동 저장**: 분석된 메시지를 로컬에 자동 저장
- **필터링**: 전체, INBOX, SPAM 별로 메시지 필터링
- **삭제 기능**: 불필요한 메시지 개별 삭제

### 3. 통계 대시보드
- **오늘 차단된 스팸 수**: 당일 필터링된 스팸 메시지 수 표시
- **전체 메시지 수**: 누적 분석된 메시지 통계
- **최근 메시지**: 최근 5개 메시지 빠른 확인

### 4. 설정 관리
- **기본 모델 선택**: 선호하는 AI 모델 설정
- **알림 설정**: 알림 활성화/비활성화
- **데이터 관리**: 전체 메시지 일괄 삭제

## 기술 스택

### 프론트엔드
- **React Native**: 크로스 플랫폼 모바일 앱 프레임워크
- **Expo SDK 54**: 개발 및 빌드 도구
- **TypeScript**: 타입 안정성
- **NativeWind 4**: Tailwind CSS 기반 스타일링
- **React 19**: 최신 React 기능 활용

### 백엔드
- **tRPC**: 타입 안전 API
- **Express**: Node.js 서버 프레임워크
- **LLM Integration**: AI 기반 메시지 분류

### 상태 관리
- **React Context API**: 전역 상태 관리
- **AsyncStorage**: 로컬 데이터 저장
- **TanStack Query**: 서버 상태 관리

## 프로젝트 구조

```
sms-filtering-app/
├── app/                      # 화면 및 라우팅
│   ├── (tabs)/              # 탭 네비게이션
│   │   ├── index.tsx        # Home 화면
│   │   ├── messages.tsx     # Messages 화면
│   │   └── settings.tsx     # Settings 화면
│   ├── analyze.tsx          # 메시지 분석 화면
│   └── _layout.tsx          # 루트 레이아웃
├── components/              # 재사용 가능한 컴포넌트
│   ├── screen-container.tsx
│   └── ui/
│       └── icon-symbol.tsx
├── lib/                     # 유틸리티 및 프로바이더
│   ├── message-provider.tsx # 메시지 상태 관리
│   ├── settings-provider.tsx # 설정 상태 관리
│   ├── storage.ts           # AsyncStorage 유틸리티
│   └── trpc.ts              # API 클라이언트
├── types/                   # TypeScript 타입 정의
│   ├── message.ts
│   └── settings.ts
├── server/                  # 백엔드 API
│   └── routers.ts           # tRPC 라우터
├── tests/                   # 테스트 파일
│   └── sms-analysis.test.ts
└── assets/                  # 이미지 및 리소스
    └── images/
        └── icon.png         # 앱 아이콘
```

## 설치 및 실행

### 사전 요구사항
- Node.js 22+
- pnpm 9+
- Expo Go 앱 (모바일 테스트용)

### 설치
```bash
cd sms-filtering-app
pnpm install
```

### 개발 서버 실행
```bash
pnpm dev
```

이 명령어는 다음을 동시에 실행합니다:
- Express 백엔드 서버 (포트 3000)
- Expo Metro Bundler (포트 8081)

### 모바일에서 테스트
1. Expo Go 앱을 모바일 기기에 설치
2. 터미널에 표시된 QR 코드를 스캔
3. 앱이 자동으로 로드됨

### 웹에서 테스트
브라우저에서 `http://localhost:8081` 접속

## 사용 방법

### 1. 메시지 분석
1. Home 화면에서 "새 메시지 분석" 버튼 클릭
2. 분석할 SMS 메시지 입력
3. (선택) AI 모델 선택 (기본값: BERT)
4. "분석하기" 버튼 클릭
5. 결과 확인 (SPAM/INBOX + 신뢰도)

### 2. 메시지 목록 확인
1. 하단 탭에서 "Messages" 선택
2. 필터 탭으로 메시지 분류 (전체/INBOX/SPAM)
3. 메시지 카드를 탭하여 상세 정보 확인
4. 스와이프 또는 삭제 버튼으로 메시지 삭제

### 3. 설정 변경
1. 하단 탭에서 "Settings" 선택
2. 기본 AI 모델 선택
3. 알림 활성화/비활성화
4. 필요시 전체 데이터 삭제

## API 엔드포인트

### POST /api/sms/analyze
SMS 메시지를 분석하여 SPAM 또는 INBOX로 분류합니다.

**요청**
```json
{
  "text": "분석할 메시지 내용",
  "model": "BERT" | "RoBERTa" | "BigBird"
}
```

**응답**
```json
{
  "classification": "SPAM" | "INBOX",
  "confidence": 0.95,
  "model": "BERT",
  "reasoning": "분류 근거"
}
```

## 테스트

```bash
# 단위 테스트 실행
pnpm test

# TypeScript 타입 체크
pnpm check
```

## 배포

### 체크포인트 생성
개발이 완료되면 체크포인트를 생성하여 버전을 저장합니다.

### 퍼블리시
UI의 "Publish" 버튼을 클릭하여 앱을 배포합니다.

## 향후 개발 계획

- [ ] 메시지 상세 화면 구현
- [ ] 재분석 기능 추가
- [ ] 데이터베이스 연동 (클라우드 동기화)
- [ ] 실제 SMS 권한 통합 (Android)
- [ ] 푸시 알림 기능
- [ ] 통계 차트 및 분석 리포트
- [ ] 사용자 정의 필터 규칙

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

## 기여

버그 리포트 및 기능 제안은 GitHub Issues를 통해 제출해주세요.

## 참고

- 기존 SMS-Filtering 프로젝트: [GitHub Repository](https://github.com/Maroco0109/SMS-Filtering)
- 사용된 모델: BERT, RoBERTa, BigBird (LLM 기반 분류)
