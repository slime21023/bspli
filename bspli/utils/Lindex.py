from .Partitioning import get_local_model_labels
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

# check if the GPU runtime env is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LIndexing:
    """
    The local learned indexing model
    To handle the query processing
    """

    def __init__(self, leafsize, epoch_num = 5, hidden_size = 100, block_range=5):
        """
        leafsize: the leaf size of local model that used by partitioning
        """
        self.leafsize = leafsize
        self.epoch_num = epoch_num
        self.hidden_size = hidden_size
        self.block_range = block_range


    def train(self, data):
        prepared_data = get_local_model_labels(
            data, leaf_size=self.leafsize)
        means, index_data, train_labels = prepared_data

        self.means = means
        self.index_data = index_data
        train_data = index_data[:, :-1]
        self.max_block = train_labels[-1, -1] + 1
        self.blocks = train_labels

        
        # Define the model for local learned index
        self.mlp = nn.Sequential(
            nn.Linear(in_features=train_data.shape[1], out_features=self.hidden_size),
            nn.LeakyReLU(negative_slope=0.09),
            nn.Linear(self.hidden_size, int(self.hidden_size/2)),
            nn.LeakyReLU(negative_slope=0.09),
            nn.Linear(int(self.hidden_size/2), self.max_block),
            # nn.Softmax(dim=1),
            nn.Hardsigmoid(),
        ).to(device)

        # print(self.mlp)
        loader = DataLoader(TensorDataset(
            train_data, train_labels), shuffle=True, batch_size=400)
        
        self.optimizer = optim.Adadelta(self.mlp.parameters(), lr=1.0, rho=0.9)
        # self.loss_fn = nn.HuberLoss(reduction ="mean", delta=0.48)
        self.loss_fn = nn.MultiMarginLoss(reduction ="sum")
        self.mlp.train()

        for epoch  in range(self.epoch_num):
            for batch, (points, labels) in enumerate(loader, 0):
                labels = labels.reshape(labels.shape[0])

                # get the inputs
                points = Variable(points.type(torch.FloatTensor))
                labels = Variable(labels.type(torch.LongTensor))

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
       

    def query(self, qp, k) -> list:
        # print(f"max blocks: {self.max_block}")
        qp = qp.reshape(1, qp.shape[0])

        # Default search blocks
        block_size = self.max_block if self.block_range > self.max_block else self.block_range
        pred_topk =  torch.topk(self.mlp(qp).flatten(), block_size).indices

        search_range = torch.where(
            torch.isin(self.blocks, pred_topk)
        )[0]
        
        search_points = self.index_data[search_range]
        norm = torch.norm(search_points[:, :-1] - qp, dim=(1))
        topk = torch.topk(norm, k, largest=False)[1]
        indices = search_points[topk, -1]

        return indices
    
