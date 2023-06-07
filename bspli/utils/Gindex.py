from .Partitioning import get_global_model_labels
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

# check if the GPU runtime env is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GIndexing:
    """
    The global learned indexing model
    To predict the query point in which local learned indexing model
    """

    def __init__(self, leafsize=2, epoch_num = 20, hidden_size = 100, block_range=2):
        self.leafsize = leafsize
        self.epoch_num = epoch_num
        self.hidden_size = hidden_size
        self.block_range = block_range

    def train(self, model_list: list):
        """
        model_list: the list of the local models
        """
        prepared_data = get_global_model_labels(model_list)
        print(f'global index train smaple count: {prepared_data.shape[0]}')
        self.index_data = prepared_data
        train_data = prepared_data[:, :-1]
        train_labels = prepared_data[:, -1]
        self.max_block = int(train_labels[-1])

        # Define the model for global learned index
        self.mlp = nn.Sequential(
            nn.Linear(in_features=train_data.shape[1], out_features=self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, int(self.hidden_size/2)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.hidden_size/2), 1),
        ).to(device)
        

        loader = DataLoader(TensorDataset(
            train_data, train_labels
        ), shuffle=True, batch_size=5)

        self.optimizer = optim.Adamax(self.mlp.parameters(), lr=0.1)
        self.loss_fn = nn.SmoothL1Loss(reduction ="mean")
        self.mlp.train()

        for epoch  in range(self.epoch_num):
            for batch, (points, labels) in enumerate(loader, 0):
                labels = labels.reshape(labels.shape[0])

                # get the inputs
                points = Variable(points.type(torch.FloatTensor))
                labels = Variable(labels.type(torch.FloatTensor))

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # predict classes using points from the training set
                outputs = self.mlp(points.to(torch.float32))

                # compute the loss based on model output and real labels
                loss = self.loss_fn(outputs, labels)

                # backpropagate the loss
                loss.backward()

                # adjust parameters based on the calculated gradients
                self.optimizer.step()

                if batch % 100 == 99:
                    print(f'{epoch + 1}, {batch + 1} loss: {loss.item() / 100 }')

        self.mlp.eval()

    def query(self, qp):
        # print(f'The global max block num: {self.max_block}')
        qp = qp.reshape(1, qp.shape[0]) 
        pred = (self.mlp(qp).int().item())
        pred = 0 if pred < 0 else pred
        pred = self.max_block if pred > self.max_block else pred
        return pred
        