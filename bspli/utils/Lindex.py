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

        # Define the model for local learned index
        self.mlp = nn.Sequential(
            nn.Linear(in_features=train_data.shape[1], out_features=self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, int(self.hidden_size/2)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(self.hidden_size/2), 1),
        ).to(device)

        # print(self.mlp)
        loader = DataLoader(TensorDataset(
            train_data, train_labels), shuffle=True, batch_size=100)
        
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
       

    def query(self, qp, k) -> list:
        qp = qp.reshape(1, qp.shape[0])
        pred =  (self.mlp(qp)[0][0]).int()
        # Default search blocks
        block_size = self.block_range

        min_block_number = 0 if pred - block_size < 0 else (pred - block_size)
        max_block_number = pred + block_size
        max_block_number = max_block_number if max_block_number < self.max_block else self.max_block

        def get_search_block(mininal, maximum, k: int):
            search_points = self.index_data[:, :-1]
            search_points = search_points[
                (search_points[:, -1] >= mininal) & 
                (search_points[:, -1] <= maximum)
            ]

            if search_points.shape[0] < k:
                mininal = mininal -1 if mininal > 0 else 0
                maximum = maximum +1 if maximum < self.max_block else self.max_block
                return get_search_block(mininal, maximum, k)
            else:
                return search_points


        search_points = get_search_block(
            mininal=min_block_number,
            maximum=max_block_number,
            k=k
        )

        # print(f"search_points shape: {search_points.shape}")

        norm = torch.norm(search_points - qp, dim=(1))
        topk = torch.topk(norm, k, largest=False)[1]
        indices = self.index_data[topk, -1]

        return indices
