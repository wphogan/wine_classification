import torch
import torch.nn as nn
import torch.functional as F

class LassoLoss(nn.Module):
    def __init__(self):
        super(LassoLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, targets, weights, lasso_lambda):
        l1_norm = torch.norm(weights, p=1, dim=1)
        l1_mean = torch.mean(l1_norm)
        loss_ce = self.cross_entropy(logits, targets)
        return loss_ce + (lasso_lambda * l1_mean)