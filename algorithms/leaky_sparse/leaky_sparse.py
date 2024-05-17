import torch
import torch.nn as nn


class LeakySparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def forward(self, X):
        k = torch.tensor(100).to(self.device)
        X_copy = X.clone()
        X_copy[X_copy < k] = torch.tanh(X_copy[X_copy < k])
        X_copy = torch.mean(X_copy)
        return X_copy
