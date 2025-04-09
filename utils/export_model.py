import os
import argparse
import torch
import yaml
from argparse import Namespace
from transformers import AutoTokenizer

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plm import LightningPLM


class TorchScriptWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits  # dict → tensor


def load_hparams_from_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ hparams.yaml not found at {path}")
    with open(path, "r") as f:
        hparams_dict = yaml.safe_load(f)
    return Namespace(**hparams_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_pt", type=str, required=True)
    parser.add_argument("--hparams_yaml", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # Load hparams
    hparams = load_hparams_from_yaml(args.hparams_yaml)
    print(f"✅ Loaded hparams from: {args.hparams_yaml}")

    # Load model
    model = LightningPLM.load_from_checkpoint(args.model_pt, hparams=hparams)
    model.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
    dummy = tokenizer("테스트 문장입니다.", return_tensors="pt", padding="max_length",
                      truncation=True, max_length=hparams.max_len)
    dummy_input = (
        dummy["input_ids"].to(model.device),
        dummy["attention_mask"].to(model.device)
    )

    # Wrap the model
    wrapper = TorchScriptWrapper(model.model)
    wrapper.eval().to(model.device)  # ✅ 모델을 GPU에 올림

    # Trace
    traced = torch.jit.trace(wrapper, dummy_input)

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "model.pt")
    traced.save(save_path)
    print(f"✅ TorchScript model saved to: {save_path}")


if __name__ == "__main__":
    main()
