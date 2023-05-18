import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, mean_reduce=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.alpha = alpha
        self.mean_reduce = mean_reduce

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logpt = logits.log_softmax(-1)
        logpt = logpt.gather(1, labels.view(-1,1)).squeeze()
        pt = logpt.exp()
        if self.alpha is not None:
            logpt = logpt * self.alpha.gather(0, labels)
        loss = -1 * ( 1 - pt ) ** self.gamma * logpt
        if self.mean_reduce: 
            return loss.mean()
        return loss.sum()


class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        if reduction and reduction not in {'mean', 'sum'}:
            raise ValueError("`reduction` must be `mean`, `sum` or None")
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        n = logits.size()[-1]
        log_preds = F.log_softmax(logits, dim=-1)
        loss = self._apply_reduction(log_preds)
        nll = F.nll_loss(
            log_preds, 
            y_true, 
            reduction=self.reduction, 
            ignore_index=self.ignore_index
        )
        return loss * self.eps / n + (1 - self.eps) * nll

    def _apply_reduction(self, log_preds: torch.Tensor) -> torch.Tensor:
        loss = -log_preds.sum(dim=-1)
        if self.reduction == 'sum':
            return -log_preds.sum()
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss

