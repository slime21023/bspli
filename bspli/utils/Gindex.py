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

    def __init__(self, epoch_num = 20, hidden_size = 100, block_range=2):
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
            # nn.Linear(in_features=train_data.shape[1], out_features=self.hidden_size),
            nn.LazyLinear(self.hidden_size, bias=False),
            nn.PReLU(),
            nn.LazyLinear(self.hidden_size, bias=False),
            nn.PReLU(),
            nn.LazyLinear(1, bias=False),
            nn.PReLU(),
        ).to(device)
        

        loader = DataLoader(TensorDataset(
            train_data, train_labels
        ), shuffle=True, batch_size=5)

        self.optimizer = optim.AdamW(self.mlp.parameters())
        self.loss_fn = nn.SmoothL1Loss(reduction ="sum")
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

                if batch % 10 == 9:
                    print(f'{epoch + 1}, {batch + 1} loss: {loss.item() / 100 }')

        self.mlp.eval()

    def query(self, qp, block_range = None) -> range:
        # print(f'The global max block num: {self.max_block}')
        qp = qp.reshape(1, qp.shape[0]) 
        pred = (self.mlp(qp).int().item())

        if block_range != None:
            min_block = pred - block_range
            min_block = 0 if  min_block < 0 else min_block

            max_block = pred + block_range
            max_block = self.max_block if max_block > self.max_block else max_block
        else:
            min_block = pred - self.block_range
            min_block = 0 if  min_block < 0 else min_block

            max_block = pred + self.block_range
            max_block = self.max_block if max_block > self.max_block else max_block
        
        if (min_block >= max_block):
            min_block = self.max_block if min_block > self.max_block else min_block
            return range(min_block, min_block +1)

        return range(min_block, max_block +1)
    