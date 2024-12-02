import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from scripts.KoBERTDataset import KoBERTDataset  # Assuming the dataset class is here
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

def load_model_and_tokenizer(model_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer

def evaluate_model(model, tokenizer, dataloader, device):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Assuming batch[3] contains labels
            texts, labels = batch[0], batch[3]

            # Ensure texts is a List[str]
            if isinstance(texts, torch.Tensor):
                texts = [str(t) for t in texts]

            # Tokenize input text
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128
            )

            # Move inputs and labels to the specified device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            labels = labels.to(device)

            # Get predictions
            outputs = model(**inputs)
            preds = torch.argmax(outputs, dim=1)

            # Collect true and predicted labels
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

        # Convert multi-dimensional true_labels to single dimension if needed
        if len(true_labels) > 0 and isinstance(true_labels[0], (list, np.ndarray)):
            true_labels = [label.argmax() for label in true_labels]

    # Debugging outputs
    print(f"True Labels (example): {true_labels[:10]}")
    print(f"Predicted Labels (example): {pred_labels[:10]}")
    print(f"Unique classes in true_labels: {set(true_labels)}")
    print(f"Unique classes in pred_labels: {set(pred_labels)}")

    unique_classes = set(true_labels + pred_labels)

    if len(unique_classes) == 1:
        print(f"Warning: Only one class ({list(unique_classes)[0]}) detected.")
        print("Cannot generate complete classification report.")
    else:
        print("Classification Report:")
        print(classification_report(true_labels, pred_labels, target_names=['ham', 'spam'], labels=[0, 1]))
        print(f"Precision: {precision_score(true_labels, pred_labels, average='binary'):.4f}")
        print(f"Recall: {recall_score(true_labels, pred_labels, average='binary'):.4f}")
        print(f"F1-Score: {f1_score(true_labels, pred_labels, average='binary'):.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./output_model"

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.to(device)

    # Load test data
    test_df = pd.read_csv("./data/test.csv")  # Adjust path as needed
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()

    # Prepare dataset and dataloader
    test_dataset = KoBERTDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer,
        max_len=128
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate model
    evaluate_model(model, tokenizer, test_loader, device)

if __name__ == "__main__":
    main()