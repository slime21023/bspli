from .Partitioning import get_local_model_labels
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import one_hot

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
            nn.LazyLinear(self.hidden_size, bias=False),
            nn.PReLU(),
            nn.LazyLinear(self.hidden_size, bias=False),
            nn.PReLU(),
            nn.LazyLinear(self.hidden_size, bias=False),
            nn.PReLU(),
            # nn.LazyLinear(self.hidden_size, bias=False),
            # nn.PReLU(),
            nn.LazyLinear(self.max_block, bias=False),
            nn.Softsign(),
        ).to(device)

        # print(self.mlp)
        # print(f"train labels shape: {train_labels.shape}")
        encoded_train_labels = one_hot(train_labels.reshape(train_labels.shape[0]))
        # print(f"encoded_train_labels: {encoded_train_labels.shape}")

        loader = DataLoader(TensorDataset(
            train_data, encoded_train_labels), shuffle=True, batch_size=200)
        
        self.optimizer = optim.AdamW(self.mlp.parameters())
        # self.optimizer = optim.Adadelta(self.mlp.parameters(), lr=1.0, rho=0.9)

        # self.loss_fn = nn.HuberLoss(reduction ="mean", delta=0.48)
        self.loss_fn = nn.CrossEntropyLoss(reduction ="mean", label_smoothing=0.25)
        self.mlp.train()

        for epoch  in range(self.epoch_num):
            for batch, (points, labels) in enumerate(loader, 0):

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
                    print(f'{epoch + 1}, {batch + 1} loss: {loss.item() / 200 }')

        self.mlp.eval()
       

    def query(self, qp, k=10, block_range=None) -> list:
        # print(f"max blocks: {self.max_block}")
        qp = qp.reshape(1, qp.shape[0])

        # Default search blocks
        if block_range != None:
            block_size = self.max_block if block_range > self.max_block else block_range
        else:
            block_size = self.max_block if self.block_range > self.max_block else self.block_range
        
        pred_topk =  torch.topk(self.mlp(qp).flatten(), block_size).indices

        search_range = torch.where(
            torch.isin(self.blocks, pred_topk)
        )[0]
        
        search_points = self.index_data[search_range]
        
        norm = torch.norm(search_points[:, :-1] - qp, dim=(1))
        topk = torch.topk(norm, k, largest=False)[1]
        datapoints = search_points[topk]

        return datapoints
    
    
    def query_with_threshold(self, qp, k=10, threshold=0.7) -> torch.Tensor | None:
        qp = qp.reshape(1, qp.shape[0])
        prob = self.mlp(qp).flatten()

        def min_max_normalize(data):
            """
            softsign range normalize
            """
            return (data + 1) / 2
        
        prob = min_max_normalize(prob)
        # print(f"prob: {prob}")
        preb_block = torch.where( prob> threshold)[0]
        # print(f"search range : {search_range}")

        if preb_block.size(dim=0) == 0:
            return None
        
        search_range = torch.where(
            torch.isin(self.blocks, preb_block)
        )[0] 

        search_points = self.index_data[search_range]
        norm = torch.norm(search_points[:, :-1] - qp, dim=(1))
        k = search_points.size(dim=0) if k > search_points.size(dim=0) else k

        topk = torch.topk(norm, k, largest=False)[1]
        datapoints = search_points[topk]
        return datapoints


    def isin(self, qp):
        qp = qp.reshape(1, qp.shape[0])
        search_points = self.index_data

        qp = qp.flatten()
        check_tensor =   (search_points[:, :-1] == qp).all(dim=1)
        return torch.any(check_tensor).item()
