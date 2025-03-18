import torch
import pandas as pd
import numpy as np
import json
import os
import glob
from transformers import AdamW, get_scheduler
from sklearn.utils.class_weight import compute_class_weight
from model import tokenizers_models
from utils import FocalLoss, calc_accuracy
from config import Config
from data_loader import prepare_data_loaders
from training_utils import EarlyStopping, MetricsTracker

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "json/config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Data preparation
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = prepare_data_loaders()
    
    # Model initialization
    tokenizer_model = tokenizers_models()
    model = tokenizer_model.model
    tokenizer = tokenizer_model.tokenizer

    # Loss function with class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1]),
        y=train_dataset.labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    loss_fn = FocalLoss(alpha=class_weights_tensor, gamma=2.0)

    # Optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': Config.WEIGHT_DECAY
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=Config.LEARNING_RATE)

    # Learning rate scheduler
    num_training_steps = len(train_loader) * Config.NUM_EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # Early stopping and metrics tracking
    early_stopping = EarlyStopping(
        patience=Config.EARLY_STOPPING_PATIENCE,
        path=os.path.join(config["model"]["output_file"], 'best_model.pt')
    )
    metrics_tracker = MetricsTracker()

    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss, train_acc = 0, 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            token_type_ids = batch[2].to(device)
            labels = batch[3].to(device)

            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_acc += calc_accuracy(outputs, labels)

        # Validation phase
        model.eval()
        val_loss, val_acc = 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                token_type_ids = batch[2].to(device)
                labels = batch[3].to(device)

                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                val_acc += calc_accuracy(outputs, labels)

        # Calculate average metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_dataset)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_dataset)

        metrics_tracker.update(avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc)
        
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        # Early stopping check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load best model and save
    model.load_state_dict(torch.load(early_stopping.path))
    output_dir = config["model"]["output_file"]
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Print best metrics
    best_metrics = metrics_tracker.get_best_metrics()
    print("\nBest Model Metrics:")
    print(f"Best Epoch: {best_metrics['best_epoch'] + 1}")
    print(f"Best Validation Loss: {best_metrics['best_val_loss']:.4f}")
    print(f"Best Validation Accuracy: {best_metrics['best_val_acc']:.4f}")

if __name__ == '__main__':
    main()