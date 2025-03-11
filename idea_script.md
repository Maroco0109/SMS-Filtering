### OCR API 적용 시점
- KoBERT 활용 스팸 분류 모델은 완성 되어 있음
- 사용자의 기기에서 SMS 텍스트를 받아서 이를 스팸/햄으로 분류하는 앱(App) 구현이 목표

1. SMS 수신 내용에 대해 OCR API 적용
- OCR이 인식할 수 있는 변조된 텍스트는 BERT Readable text로 변환됨
- 이 변환된 텍스트에 대해 Sentiment predict를 적용하면 된다

2. Sentiment Predict 적용 이후
- 스팸/햄 분류의 2차 필터 역할
    - 햄으로 분류된 메세지에 대해서만 실시
- 1차 텍스트 분류 결과가 햄으로 나타나면, 해당 SMS에 OCR API를 적용하여 텍스트 복구 시도
    - 복구된 텍스트에 대해 다시 Sentiment Predict 적용

#### 비교
1. SMS 수신 시점
- sentiment predict를 1번만 시행
- sms to text 과정에서 OCR API 적용 연산이 필요

2. sentiment predict: HAM 시점
- sentiment predict를 최악의 경우 2번 시행
    - 1차 검증 결과 spam이 나온 경우 1번 시행
    - 1차 검증 결과 ham이 나온 경우 OCR API 적용 연산 이후 2차 검증 실시
        - 2차 검증 결과에 따라 spam/ham 분류
- sms to text(단순 변환), sms with OCR(OCR 변환) 두 가지 변환 과정이 적용될 수 있음