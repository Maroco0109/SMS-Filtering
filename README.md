# Spam SMS Filtering with Natural language models

1. About Project

- NLP모델들을 활용하여 스팸 메세지를 필터링하는 모델 구현
  - 스팸 메세지에 해당하는 카테고리를 설정하여 KoBERT 모델을 통해 해당 SMS가 스팸 카테고리에 포함되는지 파악하여 필터링
  - 여러 한국어 처리 pre-trained 모델 사용 및 성능 비교 분석(https://github.com/monologg)
  - KoBERT
  - KoELECTRA
  - KoRoberta
  - KoBigBird
- Sentiment Predict Example
  - ![Sentiment Predict Example](https://github.com/user-attachments/assets/094c3de1-eddc-4d16-b66e-29129824343b)

2. Project Setup

- Model Development Environmnet
  - Anaconda3 in Windows 11
    - GPU setup for deep learning(RTX 3060 12gb)
      - CUDA
      - CuDNN
  - Python
    - Pytorch
- Project architecture
  - ![SMS Filtering](https://github.com/user-attachments/assets/511bd687-edcf-4e68-bd1f-88dc86e59242)

## 📂 Directory Structure

```bash
Spam SMS Filtering/
├── data/ # 원본 및 전처리 데이터 저장
│ ├── raw/ # 원본 데이터
│ └── preprocessed/ # 전처리된 데이터
├── preprocess/
│ ├── build_dataset.py
│ ├── merge_sms_datasets.py
│ ├── preprocess.py
│ ├── process_test.py
│ └── util.py
├── templates/
│ ├── index.html  # demo.py web page
├── utils/
│ ├── __init__.py
│ ├── convert_to_onnx.py   # pt to onnx
│ ├── data_util.py   # 데이터 토큰화
│ ├── dataset_diversity.py # dataset 분포 확인
│ ├── demo_console.py   # user input을 console에서 받는 demo
│ ├── demo.py  # user input을 web에서 받는 demo
│ ├── export_model.py   # ckpt to pt
│ ├── logger.py   # Log 파일 생성
│ ├── model_util.py  # 한국어 LLM
│ └── predict_text.py   # 감성 분석
├── main.py # 메인
├── plm.py  # 모델 호출 및 훈련
├── dataloader.py # 데이터로더
├── eval.py # 모델 평가
├── requirements.txt # 필요한 Python 패키지 리스트
├── main.py # 전체 워크플로우를 실행하는 스크립트
└── README.md # 프로젝트 설명
```

## 📋 Description

이 프로젝트는 **한국어 처리 LLM**을 활용하여 스팸 메시지와 정상 메시지를 분류하는 모델을 구축하는 것을 목표로 합니다. 데이터 전처리, 학습, 평가 과정을 통해 전체 파이프라인을 구성하며, 각각의 과정은 재사용 가능한 모듈로 구성되어 있습니다.

---

## 🚀 How to Run

1. **환경 설정**:

```bash
pip install -r requirements.txt
```

2. 데이터 전처리:

```bash
python utils/data_preprocessing.py
```

### without testcase

```bash
python build_dataset.py --split --data_dir ../data --save_dir ../result
```

### with testcase

```bash
python build_dataset.py --split --use_test --data_dir ../data --save_dir ../result
```

3. 모델 학습:

### Arguments

- model_type: bert, electra, roberta, bigbird
- model_name: bert+revised, electra+revised, roberta+revised, bigbird+revised
- max_len: 학습에 사용할 문장 길이(64, 128, 256, ...)
- gpuid: 0(default), 1(if using 2nd gpu)
- use_custom_classifier: custom classifier를 사용할 때(default: pure LLM)
- use_focal_loss: focal loss를 사용할 때(default: cross entrophy)
- threshold 0.xx: threshold 조정 시(default: 0.5)

```bash
python main.py --train --data_dir result \
--model_type {model} \
--model_name {model+revised} \
--max_len {hprams} \
--gpuid 0 \
--use_custom_classifier \
--use_focal_loss \
--threshold {0.xx}
```

4. 테스트:

### Arguments

- model_pt: 학습 완료된 ckpt 파일 경로

#### pred\_{model}+revised.csv 생성

```bash
python main.py --pred --data_dir result \
--model_type {model} \
--model_name {model+revised} \
--model_pt {model.ckpt directory} \
--max_len {hparams.yaml 참조} \
--gpuid 0 \
--use_custom_classifier \
--use_focal_loss \
--threshold=0.6
```

#### ckpt test with user inputs

```bash
python utils/predict_text.py \
--model_type {model} \
--model_pt {model.ckpt directory} \
--gpuid 0
```

### pred\_{model}+revised.csv 데이터 분포 출력

```bash
python utils/test_case.py \
--model_type {model}
```

5. 모델 추출

### ckpt to pt

```bash
python utils/export_model.py \
--model_type {model} \
--model_pt {.ckpt file directory} \
--max_len {hparams.yaml 참조} \
--output_dir {.pt file directory} \
--hparams_path {hparams.yaml directory}
```

### pt to onnx

```bash
python utils/convert_to_onnx.py \
--model_type {model}
```

## Version Issues

### Pytorch 2.7.0, Cuda 12.7

- Pytorch 2.7.0, transformers 4.46.3 에서 연산자 문제 발생
  - Scaled dot product attention

### 해결 방법

- Pytorch = 2.3.1, transformers = 4.26.1, cuda = 11.8

```bash
pip install torch==2.3.1+cu118 torchvision==0.16.1+cu118 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

```bash
pip install transformers==4.26.1

```

- 기타 패키지

```bash
pip install pytorch-lightning==2.0.9 torchmetrics==0.11.4
```
