import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Sparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def forward(self, X):
        k = torch.tensor(100).to(X.device)
        X_copy = X.clone()
        X_copy = torch.sum(X_copy, dim=0)
        X_copy[X_copy<k] = 0
        return X_copy


if __name__ == "__main__":
    x = torch.linspace(-200, 200, 100)
    x = x.reshape(1,-1)
    ss = Sparse()
    y = ss(x)
    y = y.reshape(-1)
    x = x.reshape(-1)
    plt.plot(x, y)
    plt.show()
