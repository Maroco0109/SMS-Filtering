import torch
from transformers import AutoTokenizer

# 모델 경로
model_paths = {
    "bert": "result/model/bert_bertrevised.pt",
    "electra": "result/model/electra_electrarevised.pt",
    "roberta": "result/model/roberta_robertarevised.pt"
}

# tokenizer 로드 (모델 종류에 따라 다를 수 있음, 우선 bert 기반 tokenizer 사용)
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

# 모델 로드
models = {}
for name, path in model_paths.items():
    models[name] = torch.jit.load(path, map_location='cpu')
    models[name].eval()

def predict(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)

    probs = torch.softmax(outputs, dim=-1)
    spam_prob = probs[0][0].item() * 100  # 0번 클래스: 스팸
    ham_prob = probs[0][1].item() * 100   # 1번 클래스: 햄

    return spam_prob, ham_prob

if __name__ == "__main__":
    print("✏️ 키보드 입력을 해주세요 ('exit' 입력 시 종료)")

    while True:
        user_input = input("\n> 입력 문장: ")

        if user_input.lower() == "exit":
            print("👋 프로그램을 종료합니다.")
            break

        for model_name, model in models.items():
            spam_prob, ham_prob = predict(user_input, model, tokenizer)
            print(f"\n[{model_name.upper()} 모델 결과]")
            print(f"  📌 스팸 확률: {spam_prob:.2f}%")
            print(f"  📌 햄 확률: {ham_prob:.2f}%")
