import torch
import torch.nn as nn
import torch


class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, data: torch.Tensor):
        target = torch.arange(0, data.shape[0],dtype=torch.long)
        return self.ce(data, target.to(self.device))
