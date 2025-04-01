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
├── preprocess/
│ ├── build_dataset.py
│ ├── preprocess.py
│ ├── process_test.py
│ └── util.py
├── utils/
│ ├── __init__.py
│ ├── config.py # 하이퍼파라미터 및 매개변수
│ ├── data_util.py   # 데이터 토큰화
│ ├── logger.py   # Log 파일 생성
│ ├── model_util.py  # 한국어 LLM
│ ├── data_preprocessing.py # 스팸/햄 데이터 전처리 관련 코드
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

### Electra

- Cross Entrophy

```bash
python main.py --train --data_dir result \
--model_type electra --model_name electra+revised --max_len 64 --gpuid 0
```

- Focal loss

```bash
python main.py --train --data_dir result \
--model_type electra --model_name electra+revised --max_len 64 --gpuid 0 --use_focal_loss
```

### Bert

- Cross Entrophy

```bash
python main.py --train --data_dir result \
--model_type bert --model_name bert+revised --max_len 64 --gpuid 0
```

- Focal loss

```bash
python main.py --train --data_dir result \
--model_type bert --model_name bert+revised --max_len 64 --gpuid 0 --use_focal_loss
```

### Roberta

- Cross Entrophy

```bash
python main.py --train --data_dir result \
--model_type roberta --model_name roberta+revised --max_len 64 --gpuid 0
```

- Focal loss

```bash
python main.py --train --data_dir result \
--model_type roberta --model_name roberta+revised --max_len 64 --gpuid 0 --use_focal_loss
```

### Bigbird

- Cross Entrophy

```bash
python main.py --train --data_dir result \
--model_type bigbird --model_name bigbird+revised --max_len 64 --gpuid 0
```

- Focal loss

```bash
python main.py --train --data_dir result \
--model_type bigbird --model_name bigbird+revised --max_len 64 --gpuid 0 --use_focal_loss
```

4. 테스트:

### 데이터 출력

```bash
python utils/test_case.py
```

### Bert

```bash
python main.py --pred --data_dir result \
--model_type bert --model_name bert+revised \
--model_pt model_ckpt/epoch=03-avg_val_acc=1.00.ckpt --max_len 64 --gpuid 0
```

```bash
python utils/predict_text.py --model_type bert --model_pt model_ckpt/epoch=03-avg_val_acc=1.00.ckpt --gpuid 0
```

### Electra

```bash
python main.py --pred \
--data_dir result \
--model_type electra \
--model_name electra+revised \
--model_pt model_ckpt/epoch=01-avg_val_acc=1.00.ckpt \
--max_len 64 --gpuid 0
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