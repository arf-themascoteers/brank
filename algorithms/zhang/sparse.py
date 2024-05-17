import torch
import torch.nn as nn


class Sparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def forward(self, X):
        k = torch.tensor(100).to(self.device)
        X_copy = X.clone()
        X_copy = torch.sum(X_copy, dim=0)
        X_copy[X_copy<k] = torch.tanh(X_copy[X_copy<k])
        return X_copy
