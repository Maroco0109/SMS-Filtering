import sys
import os
import argparse

# 경로 설정: 프로젝트 상위 폴더에 plm.py, utils/ 디렉토리가 있다고 가정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from utils.model_util import load_model
from utils.data_util import encode

def predict_input_pt(text, model, tokenizer, device, max_len=64):
    """
    PyTorch 체크포인트 모델로 예측
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        # Hugging Face 스타일 인코딩
        input_ids, attention_mask = encode("[CLS] " + text + " [SEP]", tokenizer, max_len=max_len)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

        # forward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        pred = torch.argmax(logits, dim=-1).item()
        return pred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='bert',
                        help='bert / electra / roberta 등 모델 이름')
    parser.add_argument('--model_pt',   type=str,
                        help='PyTorch 체크포인트 경로 (.ckpt)')
    parser.add_argument('--onnx_model', type=str,
                        help='ONNX 모델 파일 경로 (.onnx)')
    parser.add_argument('--max_len',    type=int, default=64,
                        help='토큰 최대 길이')
    parser.add_argument('--gpuid',      type=int, default=0,
                        help='GPU ID (CUDA 사용 시)')
    args = parser.parse_args()

    # ONNX 모드
    if args.onnx_model:
        import numpy as np
        import onnxruntime as ort

        # ONNX InferenceSession 생성
        session = ort.InferenceSession(args.onnx_model)
        input_names  = [inp.name for inp in session.get_inputs()]
        output_name  = session.get_outputs()[0].name

        # 토크나이저만 로드
        _, tokenizer = load_model(args.model_type, num_labels=2)

        print("📢 ONNX 모델 로드 완료! 문장을 입력하세요. 종료하려면 'exit' 입력")
        while True:
            text = input("입력 > ").strip()
            if text.lower() == 'exit':
                break

            # encode → NumPy 2D 배열(int64)
            input_ids, attention_mask = encode("[CLS] " + text + " [SEP]", tokenizer, max_len=args.max_len)
            feed = {
                input_names[0]: np.array([input_ids], dtype=np.int64),
                input_names[1]: np.array([attention_mask], dtype=np.int64),
            }

            # ONNX 추론
            logits = session.run([output_name], feed)[0]  # shape: (1, 2)
            pred   = int(np.argmax(logits, axis=-1)[0])

            label_map = {0: "Spam 📛", 1: "Ham ✅"}
            print(f"🤖 예측 결과: {label_map[pred]}")

    # PyTorch 체크포인트 모드
    else:
        from plm import LightningPLM

        # device 설정
        device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')

        # HuggingFace 모델 + 토크나이저 로드
        model, tokenizer = load_model(args.model_type, num_labels=2)
        args.num_labels = 2

        # LightningPLM에서 학습된 체크포인트 로드
        plm_model = LightningPLM.load_from_checkpoint(checkpoint_path=args.model_pt, **vars(args))
        model = plm_model.model.to(device)

        print("📢 PyTorch 체크포인트 로드 완료! 문장을 입력하세요. 종료하려면 'exit' 입력")
        while True:
            text = input("입력 > ").strip()
            if text.lower() == 'exit':
                break

            pred = predict_input_pt(text, model, tokenizer, device, max_len=args.max_len)
            label_map = {0: "Spam 📛", 1: "Ham ✅"}
            print(f"🤖 예측 결과: {label_map[pred]}")

if __name__ == '__main__':
    main()
