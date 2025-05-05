import torch
import argparse
import yaml
import os
import types
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plm import LightningPLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_pt', type=str, required=True)         # .ckpt 파일
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--hparams_path', type=str, required=True)
    args = parser.parse_args()

    # hparams.yaml 불러오기
    with open(args.hparams_path, 'r') as f:
        hparams = yaml.safe_load(f)
    if isinstance(hparams, types.SimpleNamespace):
        hparams = vars(hparams)

    # Lightning ckpt 로드
    model = LightningPLM.load_from_checkpoint(args.model_pt, hparams=hparams)
    model.eval()
    model.to('cpu')

    # HuggingFace base 모델만 추출
    plm_model = model.model

    # 저장 경로 생성
    os.makedirs(args.output_dir, exist_ok=True)
    model_filename = args.model_type + "_" + hparams["model_name"].replace("+", "") + ".pt"
    output_path = os.path.join(args.output_dir, model_filename)

    # ✅ state_dict 저장
    torch.save(plm_model.state_dict(), output_path)

    print(f"✅ Saved state_dict to {output_path}")

if __name__ == "__main__":
    main()