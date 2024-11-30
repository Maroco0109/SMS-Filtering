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

## 📂 Directory Structure

```bash
Spam SMS Filtering/
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
│ ├── init.py # 패키지 초기화 파일
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
```

## 📋 Description

이 프로젝트는 **KoBERT**를 활용하여 스팸 메시지와 정상 메시지를 분류하는 모델을 구축하는 것을 목표로 합니다. 데이터 전처리, 학습, 평가 과정을 통해 전체 파이프라인을 구성하며, 각각의 과정은 재사용 가능한 모듈로 구성되어 있습니다.

---

## 📁 Details

### **1. Data**

- **`data/raw/`**: 원본 데이터 파일이 저장되는 디렉토리입니다.
- **`data/preprocessed/`**: 전처리된 데이터 파일이 저장되는 디렉토리입니다.

### **2. Notebooks**

- 주피터 노트북 파일로 초기 분석 및 실험이 포함되어 있습니다:
  - `KoBERT_Practice.ipynb`: KoBERT 모델 학습 및 평가 실험.
  - `data_split.ipynb`: 데이터셋 분할 실험.
  - `dataset.ipynb`: 데이터 병합 실험.
  - `spam_preprocessing.ipynb`: 스팸 데이터 전처리 실험.
  - `ham_preprocessing.ipynb`: 햄 데이터 전처리 실험.
  - `ham_xlsx.ipynb`: Excel 형식의 햄 데이터 전처리 실험.

### **3. Scripts**

- 주요 기능을 Python 모듈로 분리하여 재사용성을 높였습니다:
  - **`data_split.py`**: 데이터셋을 `train`과 `test`로 분할합니다.
  - **`data_preprocessing.py`**: 스팸 및 햄 데이터를 전처리합니다.
  - **`merge_dataset.py`**: 여러 데이터 파일을 병합합니다.
  - **`model_training.py`**: KoBERT 모델 학습.
  - **`evaluation.py`**: 모델 평가.
  - **`utils.py`**: 공통적으로 사용되는 유틸리티 함수들.

### **4. Tests**

- 각 모듈에 대해 독립적인 테스트를 작성하여 코드 품질을 보장합니다.

---

## 🚀 How to Run

1. **환경 설정**:

```bash
pip install -r requirements.txt
```

2. 데이터 전처리:

```bash
python scripts/data_preprocessing.py
```

3. 데이터셋 분할:

```bash
python scripts/data_split.py
```

4. 모델 학습:

```bash
python scripts/model_training.py
```

5. 모델 평가:

```bash
python scripts/evaluation.py
```

📦 Requirements
이 프로젝트에서 필요한 Python 패키지는 requirements.txt에 명시되어 있습니다. 설치하려면 다음 명령어를 실행하세요:

```bash
pip install -r requirements.txt
```
