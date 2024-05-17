import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SSparse(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def forward(self, X):
        n = 1
        batch_size = X.shape[0]
        k = torch.tensor(100)
        X = torch.sum(X, dim=0)
        X = torch.where(X < k, (batch_size/n)* torch.tanh(X / batch_size), X)
        return X


if __name__ == "__main__":
    x = torch.linspace(-200, 200, 100)
    x = x.reshape(1,-1)
    ss = SSparse()
    y = ss(x)
    y = y.reshape(-1)
    x = x.reshape(-1)
    plt.plot(x, y)
    plt.show()
