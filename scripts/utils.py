import torch
from torch import nn

# Accuracy calculation
def calc_accuracy(preds, labels):
    _, pred_classes = preds.max(dim=1)
    return (pred_classes == labels).sum().item()

# 손실 함수 정의
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()