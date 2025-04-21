import torch
import argparse
import yaml
import os
import types
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plm import LightningPLM

class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True)
    parser.add_argument('--model_pt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--hparams_path', type=str, required=True)
    args = parser.parse_args()

    # hparams 읽기
    with open(args.hparams_path, 'r') as f:
        hparams = yaml.safe_load(f)

    if isinstance(hparams, types.SimpleNamespace):
        hparams = vars(hparams)

    model = LightningPLM.load_from_checkpoint(args.model_pt, hparams=hparams)
    model.eval()
    model.to('cpu')
    
    plm_model = model.model

    dummy = {
        'input_ids': torch.randint(0, 100, (1, hparams["max_len"]), dtype=torch.long),
        'attention_mask': torch.ones((1, hparams["max_len"]), dtype=torch.long)
    }

    wrapper = Wrapper(plm_model)
    wrapper.eval()

    if args.model_type.lower() == 'bigbird':    # bigbird 모델 구조 차이로 인해 프로젝트 제외
        scripted_model = torch.jit.script(wrapper)
    else:
        scripted_model = torch.jit.trace(wrapper, (dummy['input_ids'], dummy['attention_mask']))


    os.makedirs(args.output_dir, exist_ok=True)
    model_filename = args.model_type + "_" + hparams["model_name"].replace("+", "") + ".pt"
    output_path = os.path.join(args.output_dir, model_filename)

    scripted_model.save(output_path)
    print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    main()