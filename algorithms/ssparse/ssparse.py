import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SSparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def forward(self, X):
        k = torch.tensor(100/128).to(X.device)
        X_copy = X.clone()
        X_copy[X_copy<k] = torch.tanh(0.1*X_copy[X_copy<k])
        X_copy = torch.sum(X_copy, dim=0)
        return X_copy


if __name__ == "__main__":
    x = torch.linspace(-2, 2, 100)
    x = x.reshape(1,-1)
    ss = SSparse()
    y = ss(x)
    y = y.reshape(-1)
    x = x.reshape(-1)
    plt.plot(x, y)
    plt.show()
