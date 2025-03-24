```bash
cd preprocessing/
```

### 1. Build Training, Validation, Test dataset

```bash
python build_dataset.py --split --data_dir ../data --save_dir ../result
```

```bash
python build_dataset.py --split --use_test --data_dir ../data --save_dir ../result
```

### 2. Model Training

```bash
python main.py --train --data_dir result \
--model_type electra --model_name electra+revised --max_len 64 --gpuid 0
```

### 3. Test

```bash
python main.py --pred --data_dir result \
--model_type bert --model_name bert+revised \
--model_pt model_ckpt/epoch=03-avg_val_acc=1.00.ckpt --max_len 64 --gpuid 0
```

```bash
python utils/predict_text.py --model_type bert --model_pt model_ckpt/epoch=03-avg_val_acc=1.00.ckpt --gpuid 0
```

### Fine Tuning

- pretrained 모델 활용 방법
- 레이어 추가
-
