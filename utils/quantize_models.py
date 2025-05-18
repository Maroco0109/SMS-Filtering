from onnxruntime.quantization import quantize_dynamic, QuantType
import os

model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../result/onnx"))
model_files = ["bert.onnx", "electra.onnx", "roberta.onnx"]

output_dir = os.path.join(model_dir, "quantized")
os.makedirs(output_dir, exist_ok=True)

for model_file in model_files:
    input_path = os.path.join(model_dir, model_file)
    output_path = os.path.join(output_dir, model_file.replace(".onnx", "_quant.onnx"))

    print(f"양자화 중: {input_path} → {output_path}")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8  # ← 여기 수정
    )

print("모든 모델 양자화 완료")
