import torch
import numpy as np
from pathlib import Path

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.path)

class MetricsTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def update(self, train_loss, train_acc, val_loss, val_acc):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)

    def get_best_metrics(self):
        best_epoch = np.argmin(self.val_losses)
        return {
            'best_epoch': best_epoch,
            'best_val_loss': self.val_losses[best_epoch],
            'best_val_acc': self.val_accuracies[best_epoch],
            'corresponding_train_loss': self.train_losses[best_epoch],
            'corresponding_train_acc': self.train_accuracies[best_epoch]
        } 