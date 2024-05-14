from algorithm import Algorithm
from algorithms.reconstructor.rec_ann import RecANN
from ds_manager import DSManager
import torch
from torch.utils.data import TensorDataset, DataLoader


class AlgorithmReconstructor(Algorithm):
    def __init__(self, target_size, splits, repeat, fold):
        super().__init__(target_size, splits, repeat, fold)
        self.task = DSManager.get_task_by_dataset_name(splits.get_name())
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        torch.backends.cudnn.deterministic = True

    def get_selected_indices(self):
        recann = RecANN(self.splits.train_x.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(recann.parameters(), lr=0.001)
        X_train = torch.tensor(self.splits.train_x, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_train, X_train)
        dataloader = DataLoader(dataset, batch_size=64000, shuffle=True)
        loss = 0
        sorted_indices = []
        for epoch in range(1000):
            for batch_idx, (X, y) in enumerate(dataloader):
                optimizer.zero_grad()
                y_hat = recann(X)
                loss = self.criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                ind_loss = torch.mean(torch.pow(y - y_hat,2),dim=0)
                sorted_indices = (torch.argsort(ind_loss, descending=True)).tolist()
                print(sorted_indices[0:10])
            print(f"Epoch={epoch} Loss={round(loss.item(), 5)}")
        super()._set_all_indices(sorted_indices)
        selected_indices = sorted_indices[: self.target_size]
        return recann, selected_indices

    def get_name(self):
        return "rec"