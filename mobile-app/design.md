# SMS Filtering App - Design Document

## 앱 개요

BERT, RoBERTa, BigBird 모델을 활용하여 스팸 SMS를 자동으로 필터링하는 모바일 앱입니다. 사용자가 메시지를 입력하거나 선택하면 AI 모델이 자동으로 SPAM 또는 INBOX로 분류하고 저장합니다.

## 디자인 원칙

- **모바일 세로 모드 (9:16)** 최적화
- **한 손 사용** 가능한 UI
- **iOS HIG 준수** - 네이티브 iOS 앱과 동일한 느낌
- **간결하고 직관적인 인터페이스**

## 화면 구조

### 1. Home (메인 화면)
**주요 콘텐츠:**
- 최근 필터링된 메시지 목록 (최대 5개)
- 빠른 통계 카드 (오늘 필터링된 스팸 수, 전체 메시지 수)
- "새 메시지 분석" 버튼 (Primary CTA)

**기능:**
- 메시지 목록 탭하여 상세 보기
- 새 메시지 분석 화면으로 이동

**레이아웃:**
- 상단: 통계 카드 (2개 가로 배치)
- 중단: 최근 메시지 리스트
- 하단: Floating Action Button (새 메시지 분석)

### 2. Analyze (메시지 분석 화면)
**주요 콘텐츠:**
- 텍스트 입력 영역 (멀티라인)
- 모델 선택 드롭다운 (BERT, RoBERTa, BigBird)
- "분석하기" 버튼
- 분석 결과 표시 영역 (SPAM/INBOX 배지, 신뢰도 점수)

**기능:**
- 사용자가 SMS 텍스트 입력
- 모델 선택 (기본값: BERT)
- 분석 실행 후 결과 표시
- 결과를 데이터베이스에 저장

**레이아웃:**
- 상단: 모델 선택 드롭다운
- 중단: 텍스트 입력 영역 (높이 자동 조절)
- 하단: 분석 버튼 + 결과 카드

### 3. Messages (전체 메시지 목록)
**주요 콘텐츠:**
- 탭 필터 (전체 / INBOX / SPAM)
- 메시지 리스트 (무한 스크롤)
- 각 메시지 카드: 텍스트 미리보기, 분류 배지, 날짜

**기능:**
- 탭으로 필터링
- 메시지 탭하여 상세 보기
- 스와이프로 삭제

**레이아웃:**
- 상단: 탭 바 (전체/INBOX/SPAM)
- 하단: FlatList로 메시지 카드 렌더링

### 4. Detail (메시지 상세)
**주요 콘텐츠:**
- 전체 메시지 텍스트
- 분류 결과 (SPAM/INBOX)
- 신뢰도 점수
- 사용된 모델 정보
- 분석 날짜/시간

**기능:**
- 재분석 버튼 (다른 모델로)
- 삭제 버튼

**레이아웃:**
- 스크롤 가능한 단일 카드 형식
- 상단: 분류 배지 + 신뢰도
- 중단: 메시지 전문
- 하단: 메타데이터 + 액션 버튼

### 5. Settings (설정)
**주요 콘텐츠:**
- 기본 모델 선택
- 알림 설정
- 데이터 관리 (전체 삭제)
- 앱 정보 (버전, 라이선스)

**기능:**
- 설정 변경 및 저장
- 데이터 초기화

**레이아웃:**
- 섹션별 그룹화된 설정 항목
- iOS Settings 앱 스타일

## 주요 사용자 플로우

### Flow 1: 새 메시지 분석
1. Home 화면에서 "새 메시지 분석" 버튼 탭
2. Analyze 화면으로 이동
3. 메시지 텍스트 입력
4. (선택) 모델 변경
5. "분석하기" 버튼 탭
6. 로딩 인디케이터 표시
7. 결과 표시 (SPAM/INBOX + 신뢰도)
8. 자동으로 데이터베이스 저장
9. "완료" 버튼으로 Home 복귀

### Flow 2: 메시지 목록 확인
1. 하단 탭에서 "Messages" 탭 선택
2. 필터 탭 선택 (전체/INBOX/SPAM)
3. 메시지 카드 탭하여 상세 보기
4. Detail 화면에서 전체 내용 확인
5. (선택) 재분석 또는 삭제
6. 뒤로가기로 목록 복귀

### Flow 3: 통계 확인
1. Home 화면 상단 통계 카드 확인
2. 오늘 필터링된 스팸 수
3. 전체 메시지 수
4. (선택) 카드 탭하여 Messages 화면으로 이동

## 색상 선택

**Primary Color:** `#0a7ea4` (청록색) - 신뢰감과 기술적 느낌
**Success Color:** `#22C55E` (녹색) - INBOX 메시지
**Error Color:** `#EF4444` (빨간색) - SPAM 메시지
**Background:** 
- Light: `#ffffff`
- Dark: `#151718`
**Surface:**
- Light: `#f5f5f5`
- Dark: `#1e2022`

## 아이콘 매핑

- `house.fill` → Home 탭
- `tray.fill` → Messages 탭
- `gear` → Settings 탭
- `plus.circle.fill` → 새 메시지 분석 버튼
- `checkmark.shield.fill` → INBOX 배지
- `xmark.shield.fill` → SPAM 배지

## 데이터 모델

### Message
- `id`: string (UUID)
- `text`: string (메시지 내용)
- `classification`: "INBOX" | "SPAM"
- `confidence`: number (0-1)
- `model`: "BERT" | "RoBERTa" | "BigBird"
- `createdAt`: Date
- `updatedAt`: Date

### Settings
- `defaultModel`: "BERT" | "RoBERTa" | "BigBird"
- `notificationsEnabled`: boolean

## 기술 스택

- **프론트엔드:** React Native + Expo
- **스타일링:** NativeWind (Tailwind CSS)
- **상태 관리:** React Context + AsyncStorage
- **데이터베이스:** AsyncStorage (로컬 저장)
- **ML 모델:** 서버 API 호출 (기존 PyTorch 모델 활용)

## 구현 우선순위

1. **Phase 1 (MVP):**
   - Home 화면 (통계 + 최근 메시지)
   - Analyze 화면 (텍스트 입력 + 분석)
   - Messages 화면 (목록 + 필터)
   - 로컬 저장 (AsyncStorage)

2. **Phase 2:**
   - Detail 화면 (상세 보기)
   - Settings 화면
   - 재분석 기능

3. **Phase 3 (선택):**
   - 서버 연동 (실제 ML 모델 API)
   - 푸시 알림
   - 데이터 동기화
