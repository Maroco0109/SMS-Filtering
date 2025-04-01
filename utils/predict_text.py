import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
from utils.model_util import load_model
from utils.data_util import encode

def predict_input(text, model, tokenizer, device, max_len=64):
    model.eval()
    model.to(device)

    with torch.no_grad():
        input_ids, attention_mask = encode("[CLS] " + text + " [SEP]", tokenizer, max_len=max_len)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        pred = torch.argmax(logits, dim=-1).item()
        
        return pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='bert')
    parser.add_argument('--model_pt', type=str, required=True)
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--gpuid', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_model(args.model_type, num_labels=2)

    args.num_labels=2
    
    # ⚡ plm.py는 홈 디렉토리에 있으므로, 경로 그대로 유지
    from plm import LightningPLM
    plm_model = LightningPLM.load_from_checkpoint(checkpoint_path=args.model_pt, hparams=args)
    model = plm_model.model  # huggingface model

    print("📢 모델 로딩 완료! 문장을 입력하세요. 종료하려면 'exit' 입력")

    while True:
        text = input("입력 > ")
        if text.strip().lower() == 'exit':
            break

        label_map = {0: "Spam 📛", 1: "Ham ✅"}
        pred = predict_input(text, model, tokenizer, device, max_len=args.max_len)
        print(f"🤖 예측 결과: {label_map[pred]}")