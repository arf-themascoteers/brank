import torch
import matplotlib.pyplot as plt

X = torch.linspace(1,0,128)
X = X.reshape(1,-1)
X = X.repeat(128,1)

k = torch.tensor(100)
X = torch.sum(X, dim=0)
X = torch.where(X < k, 128*torch.tanh(X/128), X)

torch.set_printoptions(sci_mode=False)
print(X)

