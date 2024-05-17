from algorithm import Algorithm
import torch
from torch.utils.data import TensorDataset, DataLoader
from algorithms.leaky_sparse.leaky_sparse import LeakySparse
from algorithms.leaky_sparse.leaky_sparse_net import LeakySparseNet
import numpy as np
import math


class AlgorithmLeakySparse(Algorithm):
    def __init__(self, target_size, splits):
        super().__init__(target_size, splits)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.sparse = LeakySparse()

    def get_selected_indices(self):
        class_size = len(np.unique(self.splits.train_y))
        last_layer_input = 100
        if self.splits.get_name() == "ghisaconus":
            last_layer_input = 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        leaky_sparse_net = LeakySparseNet(self.splits.train_x.shape[1], class_size, last_layer_input).to(device)
        optimizer = torch.optim.Adam(leaky_sparse_net.parameters(), lr=0.001, betas=(0.9,0.999))
        X_train = torch.tensor(self.splits.train_x, dtype=torch.float32).to(device)
        y_train = torch.tensor(self.splits.train_y, dtype=torch.int32).to(device)
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        channel_weights = None
        loss = 0
        l1_loss = 0
        mse_loss = 0
        # y = y.type(torch.LongTensor).to(self.device)
        # y_validation = y_validation.type(torch.LongTensor).to(self.device)

        for epoch in range(500):
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                channel_weights, y_hat = leaky_sparse_net(X)
                y = y.type(torch.LongTensor).to(device)
                mse_loss = self.criterion(y_hat, y)
                sparse_loss = self.sparse(channel_weights)
                lambda_value = self.get_lambda(epoch+1)
                loss = mse_loss + lambda_value*sparse_loss
                loss.backward()
                optimizer.step()
            print(f"Epoch={epoch} MSE={round(mse_loss.item(), 5)}, Sparse={round(sparse_loss.item(), 5)}, Lambda={lambda_value}, LOSS={round(loss.item(), 5)}")
        mean_weight = torch.mean(channel_weights, dim=0)
        band_indx = (torch.argsort(mean_weight, descending=True)).tolist()
        super()._set_all_indices(band_indx)
        selected_indices = band_indx[: self.target_size]
        return leaky_sparse_net, selected_indices

    def get_name(self):
        return "zhang"

    def get_lambda(self, epoch):
        return 0.001 * math.exp(epoch/500)