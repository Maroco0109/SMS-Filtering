# Spam SMS Filtering with Natural language model and Image processing

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
├── notebooks/ # 주피터 노트북 파일 저장(old)
│ ├── KoBERT_Practice.ipynb
│ ├── data_split.ipynb
│ ├── dataset.ipynb
│ ├── spam_preprocessing.ipynb
│ ├── ham_preprocessing.ipynb
│ └── ham_xlsx.ipynb
├── scripts/ # 주요 기능을 담당하는 Python 모듈(old)
│ ├── init.py # 패키지 초기화 파일
│ ├── config.py # 하이퍼파라미터 및 매개변수
│ ├── data_loader.py # 데이터셋을 데이터 로더 형태로 변환
│ ├── data_split.py # 데이터셋 분할 관련 코드
│ ├── data_preprocessing.py # 스팸/햄 데이터 전처리 관련 코드
│ ├── merge_dataset.py # 데이터 병합 관련 코드
│ ├── model_training.py # KoBERT 모델 학습 코드
│ ├── model.py # BERT 모델 구현
│ ├── sentiment_predict.py # 감성추론 구현
│ ├── evaluation.py # 모델 평가 관련 코드
│ └── utils.py # 공통 유틸리티 함수
├── preprocess/
│ ├── build_dataset.py
│ ├── preprocess.py
│ ├── process_test.py
│ └── util.py
├── utils/
│ ├── __init__.py
│ ├── data_util.py   # 데이터 토큰화
│ ├── logger.py   # Log 파일 생성
│ ├── model_util.py  # 한국어 LLM
│ └── predict_text.py   # 감성 분석
├── main.py # 메인
├── plm.py  # 모델 호출 및 훈련
├── dataloader.py # 데이터로더
├── eval.py # 모델 평가
├── commands.md   # preprocess, build, train 명령어
├── requirements.txt # 필요한 Python 패키지 리스트
├── main.py # 전체 워크플로우를 실행하는 스크립트
└── README.md # 프로젝트 설명
```

## 📋 Description

이 프로젝트는 **한국어 처리 LLM**을 활용하여 스팸 메시지와 정상 메시지를 분류하는 모델을 구축하는 것을 목표로 합니다. 데이터 전처리, 학습, 평가 과정을 통해 전체 파이프라인을 구성하며, 각각의 과정은 재사용 가능한 모듈로 구성되어 있습니다.

---

## 📁 Details

### **1. Data**

- **`data/raw/`**: 원본 데이터 파일이 저장되는 디렉토리입니다.
- **`data/preprocessed/`**: 전처리된 데이터 파일이 저장되는 디렉토리입니다.

### **2. Notebooks(old)**

- 주피터 노트북 파일로 초기 분석 및 실험이 포함되어 있습니다:
  - `KoBERT_Practice.ipynb`: KoBERT 모델 학습 및 평가 실험.
  - `data_split.ipynb`: 데이터셋 분할 실험.
  - `dataset.ipynb`: 데이터 병합 실험.
  - `spam_preprocessing.ipynb`: 스팸 데이터 전처리 실험.
  - `ham_preprocessing.ipynb`: 햄 데이터 전처리 실험.
  - `ham_xlsx.ipynb`: Excel 형식의 햄 데이터 전처리 실험.

### **3. Scripts(old)**

- 주요 기능을 Python 모듈로 분리하여 재사용성을 높였습니다:
  - **`data_split.py`**: 데이터셋을 `train`과 `test`로 분할합니다.
  - **`data_preprocessing.py`**: 스팸 및 햄 데이터를 전처리합니다.
  - **`merge_dataset.py`**: 여러 데이터 파일을 병합합니다.
  - **`model_training.py`**: KoBERT 모델 학습.
  - **`evaluation.py`**: 모델 평가.
  - **`utils.py`**: 공통적으로 사용되는 유틸리티 함수들.

---

## 🚀 How to Run

1. **환경 설정(old)**:

```bash
pip install -r requirements.txt
```

2. 데이터 전처리:

```bash
python utils/data_preprocessing.py
```
```bash
python build_dataset.py --split --data_dir ../data --save_dir ../result
```
```bash
python build_dataset.py --split --use_test --data_dir ../data --save_dir ../result
```

3. 모델 학습:

# Electra
```bash
python main.py --train --data_dir result \
--model_type electra --model_name electra+revised --max_len 64 --gpuid 0
```

# Bert
```bash
python main.py --train --data_dir result \
--model_type bert --model_name bert+revised --max_len 64 --gpuid 0
```

# Roberta
```bash
python main.py --train --data_dir result \
--model_type roberta --model_name roberta+revised --max_len 64 --gpuid 0
```

# Bigbird
```bash
python main.py --train --data_dir result \
--model_type bigbird --model_name bigbird+revised --max_len 64 --gpuid 0
```

4. 테스트:

```bash
python main.py --pred --data_dir result \
--model_type bert --model_name bert+revised \
--model_pt model_ckpt/epoch=03-avg_val_acc=1.00.ckpt --max_len 64 --gpuid 0
```

```bash
python utils/predict_text.py --model_type bert --model_pt model_ckpt/epoch=03-avg_val_acc=1.00.ckpt --gpuid 0
```

📦 Requirements
이 프로젝트에서 필요한 Python 패키지는 requirements.txt에 명시되어 있습니다. 설치하려면 다음 명령어를 실행하세요:

```bash
pip install -r requirements.txt
```

Phase 1: Model Conversion and Optimization
1. Convert PyTorch Model to Mobile Format
   - Convert the KoBERT model to TorchScript format
   - Optimize the model size using quantization
   - Export the model in a format compatible with PyTorch Mobile

2. Prepare Model Assets
   - Export the tokenizer vocabulary and configurations
   - Create a lightweight version of the model suitable for mobile devices
   - Package model files and assets for Android integration

Phase 2: Android App Development
1. Project Setup
   - Create a new Android project in Android Studio
   - Set up the required dependencies:
     ```gradle
     dependencies {
         implementation 'org.pytorch:pytorch_android:1.13.0'
         implementation 'org.pytorch:pytorch_android_torchvision:1.13.0'
         // Add other necessary dependencies
     }
     ```

2. App Architecture(example)
   - by Kotlin or Dart
   ```
   app/
   ├── java/
   │   ├── activities/
   │   │   ├── MainActivity.java
   │   │   └── ResultActivity.java
   │   ├── model/
   │   │   ├── SpamPredictor.java
   │   │   └── KoBertTokenizer.java
   │   └── utils/
   │       └── TextPreprocessor.java
   ├── assets/
   │   ├── spam_model.pt
   │   └── vocab.txt
   └── res/
       └── layout/
           ├── activity_main.xml
           └── activity_result.xml
   ```

3. Core Features Implementation
   - Create text input interface
   - Implement real-time SMS monitoring (optional)
   - Build prediction result display
   - Add message history feature

Phase 3: Model Integration
1. Port the Prediction Logic
```java
public class SpamPredictor {
    private Module model;
    private KoBertTokenizer tokenizer;

    public SpamPredictor(Context context) {
        // Load the model
        model = Module.load(assetFilePath(context, "spam_model.pt"));
        tokenizer = new KoBertTokenizer(context);
    }

    public PredictionResult predict(String text) {
        // Tokenize and predict
        // Return spam probability and classification
    }
}
```

2. Implement Tokenizer
```java
public class KoBertTokenizer {
    // Port the Python tokenizer logic to Java
    // Implement necessary preprocessing
    public IValue tokenize(String text) {
        // Convert text to model input format
    }
}
```

Phase 4: User Interface Development
1. Main Screen
   - Text input field
   - "Check Message" button
   - History of recent checks
   - Settings button

2. Result Screen
   - Classification result (Spam/Ham)
   - Confidence score
   - Detailed analysis
   - Action buttons (Report false positive/negative)

Phase 5: Additional Features
1. Real-time SMS Monitoring
   - Background service for SMS monitoring
   - Notification system for spam detection
   - Auto-categorization of messages

2. Settings and Customization
   - Sensitivity adjustment
   - Notification preferences
   - Language settings
   - Theme options

Phase 6: Testing and Optimization
1. Performance Testing
   - Model inference speed
   - Memory usage
   - Battery consumption

2. Accuracy Testing
   - Test with various message types
   - Validate Korean language handling
   - Check edge cases

Phase 7: Deployment and Maintenance
1. Release Preparation
   - App optimization
   - Documentation
   - Privacy policy
   - Play Store listing preparation

2. Post-Release
   - Monitor performance
   - Gather user feedback
   - Plan updates and improvements

Timeline Estimation:
- Phase 1: 1-2 weeks
- Phase 2: 1 week
- Phase 3: 2 weeks
- Phase 4: 1 week
- Phase 5: 2 weeks
- Phase 6: 1 week
- Phase 7: 1 week

Total estimated time: 9-10 weeks