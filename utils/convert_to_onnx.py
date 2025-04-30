import torch
import argparse
import os
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def convert_model_to_onnx(model_type):
    # 1. 모델명 맵핑
    model_map = {
        "bert": "monologg/kobert",
        "electra": "monologg/koelectra-base-v3-discriminator",
        "roberta": "klue/roberta-base"
    }

    if model_type not in model_map:
        raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")

    model_name = model_map[model_type]

    # 2. 경로 설정
    hparam_path = os.path.join("lightning_logs", model_type, "hparams.yaml")
    pt_path = os.path.join("result", "model", f"{model_type}_{model_type}revised.pt")
    output_path = os.path.join("result", "onnx", f"{model_type}.onnx")

    if not os.path.isfile(hparam_path):
        raise FileNotFoundError(f"하이퍼파라미터 파일이 없습니다: {hparam_path}")
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"모델 .pt 파일이 없습니다: {pt_path}")

    # 3. 하이퍼파라미터 로딩
    with open(hparam_path, 'r') as f:
        hparams = yaml.safe_load(f)
    max_len = hparams.get("max_len", 128)

    # 4. 모델 및 토크나이저 로딩
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.load_state_dict(torch.load(pt_path, map_location=torch.device('cpu')))
    model.eval()

    # 5. 더미 입력
    dummy = tokenizer("이것은 스팸 메시지입니다.", return_tensors="pt", padding="max_length", max_length=max_len, truncation=True)

    # 6. ONNX 저장 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 7. 변환
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        },
        opset_version=13
    )

    print(f"✅ {model_type.upper()} → ONNX 저장 완료: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', required=True, choices=['bert', 'electra', 'roberta'])
    args = parser.parse_args()

    convert_model_to_onnx(args.model_type)

if __name__ == "__main__":
    main()
