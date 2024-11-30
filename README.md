# Spam SMS Filtering with Natural language model and Image processing

1. About Project

- NLP(KoBERT)와 Image processing(YOLO, OCR API)를 활용하여 스팸 메세지를 필터링하는 모델 구현
  - 스팸 메세지에 해당하는 카테고리를 설정하여 KoBERT 모델을 통해 해당 SMS가 스팸 카테고리에 포함되는지 파악하여 필터링
  - Image processing 기법을 통해 왜곡되어있거나 세로로 적히는 등 일반적인 Text filed에서 처리하기 힘든 SMS에 대하여 해당 텍스트를 복구한 후, 위의 모델을 통해 스펨 카테고리에 포함되는지 파악하여 필터링
- Sentiment Predict Example
  - ![Sentiment Predict Example](https://github.com/user-attachments/assets/094c3de1-eddc-4d16-b66e-29129824343b)

2. Project Setup

- Model Development Environmnet
  - Anaconda3 in Windows 11
    - GPU setup for deep learning(RTX 3060 12gb)
      - CUDA
      - CuDNN
  - Python
    - ver 3.7.11
    - packages
      - Tensorflow
        - Keras
      - KoBERT
      - YOLO
      - OCR API(Naver Clova, Google Cloud Vision)
- Project architecture
  - ![SMS Filtering](https://github.com/user-attachments/assets/511bd687-edcf-4e68-bd1f-88dc86e59242)
  - project_name/
    ├── data/ # 원본 및 전처리 데이터 저장
    │ ├── raw/ # 원본 데이터
    │ └── preprocessed/ # 전처리된 데이터
    ├── notebooks/ # 주피터 노트북 파일 저장
    │ ├── KoBERT_Practice.ipynb
    │ ├── data_split.ipynb
    │ ├── dataset.ipynb
    │ ├── spam_preprocessing.ipynb
    │ ├── ham_preprocessing.ipynb
    │ └── ham_xlsx.ipynb
    ├── scripts/ # 주요 기능을 담당하는 Python 모듈
    │ ├── **init**.py # 패키지 초기화 파일
    │ ├── data_split.py # 데이터셋 분할 관련 코드
    │ ├── data_preprocessing.py # 스팸/햄 데이터 전처리 관련 코드
    │ ├── merge_dataset.py # 데이터 병합 관련 코드
    │ ├── model_training.py # KoBERT 모델 학습 코드
    │ ├── evaluation.py # 모델 평가 관련 코드
    │ └── utils.py # 공통 유틸리티 함수
    ├── tests/ # 테스트 코드
    │ ├── test_data_preprocessing.py # 데이터 전처리 테스트
    │ ├── test_model_training.py # 모델 학습 테스트
    │ └── test_merge_dataset.py # 데이터 병합 테스트
    ├── requirements.txt # 필요한 Python 패키지 리스트
    ├── main.py # 전체 워크플로우를 실행하는 스크립트
    └── README.md # 프로젝트 설명
