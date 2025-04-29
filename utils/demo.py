# utils/demo.py
from flask import Flask, render_template, request
import torch
import os
from transformers import AutoTokenizer

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)

# templates 경로 지정
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates_dir = os.path.join(base_dir, 'templates')

app = Flask(__name__, template_folder=templates_dir)

# 모델 로드
model_paths = {
    'bert': 'result/model/bert_bertrevised.pt',
    'electra': 'result/model/electra_electrarevised.pt',
    'roberta': 'result/model/roberta_robertarevised.pt'
}
models = {name: torch.jit.load(path, map_location='cpu').eval() for name, path in model_paths.items()}

# 기록 저장
history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        results = {}
        for name, model in models.items():
            with torch.no_grad():
                outputs = model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)[0]
                spam_prob = round(probs[0].item() * 100, 2)
                ham_prob = round(probs[1].item() * 100, 2)
                results[name] = {"ham": ham_prob, "spam": spam_prob}
        
        # 기록 저장
        history.append({
            'text': text,
            'prediction': results
        })

        return render_template('index.html', prediction=results, history=history)

    # GET 방식이면 빈 prediction과 history 넘기기
    return render_template('index.html', prediction=None, history=history)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
