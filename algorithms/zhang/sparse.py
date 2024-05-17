import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Sparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def forward(self, X):
        k = torch.tensor(100).to(X.device)
        X = torch.sum(X, dim=0)
        X = torch.where(X < k, 0, X)
        return X

