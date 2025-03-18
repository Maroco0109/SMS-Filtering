import torch
import os
from model import BERTClassifier, tokenizers_models
from transformers import AutoTokenizer

def export_model_for_mobile(model_path, output_dir):
    # Load the tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer_model = tokenizers_models()
    model = tokenizer_model.model
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Example input for tracing
    example_input_ids = torch.randint(1000, (1, 128))  # batch_size=1, seq_length=128
    example_attention_mask = torch.ones((1, 128))
    example_token_type_ids = torch.zeros((1, 128))
    
    # Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(model, (example_input_ids, example_attention_mask, example_token_type_ids))
    
    # Optimize the traced model
    print("Optimizing model...")
    traced_model_optimized = torch.jit.optimize_for_inference(traced_model)
    
    # Save the optimized model
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "model_mobile.pt")
    traced_model_optimized.save(model_save_path)
    print(f"Mobile-optimized model saved to: {model_save_path}")
    
    # Save tokenizer files
    print("Saving tokenizer files...")
    tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
    tokenizer_save_path = os.path.join(output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"Tokenizer saved to: {tokenizer_save_path}")

def quantize_model(model_path, output_path):
    """
    Quantize the model to reduce its size (optional)
    """
    # Load the traced model
    model = torch.jit.load(model_path)
    
    # Quantize the model to int8
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Specify which layers to quantize
        dtype=torch.qint8
    )
    
    # Save the quantized model
    quantized_model.save(output_path)
    print(f"Quantized model saved to: {output_path}")

if __name__ == "__main__":
    # Paths configuration
    trained_model_path = "path/to/your/trained/model.pt"  #### Update this path
    mobile_export_dir = "mobile_model"
    
    # Export model for mobile
    export_model_for_mobile(trained_model_path, mobile_export_dir)
    
    # Optionally quantize the model
    quantize_model(
        os.path.join(mobile_export_dir, "model_mobile.pt"),
        os.path.join(mobile_export_dir, "model_mobile_quantized.pt")
    ) 