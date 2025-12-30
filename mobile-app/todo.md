# SMS Filtering App - TODO

## PyTorch 모델 통합 작업 (Python FastAPI)
- [x] Python FastAPI 모델 서버 구축
- [x] Hugging Face Transformers 통합
- [x] ML 클라이언트 TypeScript 구현
- [x] API 라우터 업데이트 (PyTorch + LLM 폴백)
- [x] 헬스 체크 및 에러 핸들링
- [x] 문서 작성
- [ ] 실제 테스트 (모델 서버 실행 필요)
- [ ] GitHub 업데이트

## Phase 1: 기본 구조 및 UI 구현
- [x] 앱 브랜딩 (로고 생성 및 app.config.ts 업데이트)
- [x] 탭 네비게이션 구조 설정 (Home, Messages, Settings)
- [x] 아이콘 매핑 추가 (icon-symbol.tsx)
- [x] 테마 색상 커스터마이징
- [x] Home 화면 레이아웃 구현
- [x] Analyze 화면 레이아웃 구현
- [x] Messages 화면 레이아웃 구현
- [x] Settings 화면 레이아웃 구현

## Phase 2: 데이터 모델 및 상태 관리
- [x] Message 타입 정의
- [x] Settings 타입 정의
- [x] AsyncStorage 유틸리티 함수 작성
- [x] Context Provider 구현 (MessageContext)
- [x] Context Provider 구현 (SettingsContext)

## Phase 3: ML 모델 통합
- [x] 서버 API 엔드포인트 설계 (SMS 분석)
- [x] 모델 추론 로직 구현 (LLM 기반 분류)
- [x] API 클라이언트 함수 작성
- [x] 분석 결과 처리 로직

## Phase 4: 핵심 기능 구현
- [x] 새 메시지 분석 기능
- [x] 메시지 목록 필터링 (전체/INBOX/SPAM)
- [ ] 메시지 상세 보기 (추후 구현)
- [x] 메시지 삭제 기능
- [ ] 재분석 기능 (추후 구현)
- [x] 통계 계산 및 표시

## Phase 5: 데이터베이스 연동
- [x] 로컬 저장소 (AsyncStorage) 사용
- [ ] 데이터베이스 연동 (선택 사항)

## Phase 6: 테스트 및 최적화
- [x] 단위 테스트 작성
- [x] 통합 테스트 실행
- [x] 기본 에러 핸들링 구현

## Phase 7: 최종 점검
- [x] 모든 기능 테스트
- [x] UI/UX 구현 완료
- [x] 문서 업데이트
- [x] 체크포인트 생성
