import partitioning
import model
import torch
from torch.utils.data import DataLoader, TensorDataset


class LIndexing:
    """
    The local learned indexing model
    To handle the query processing
    """

    def __init__(self, leafsize):
        """
        leafsize: the leaf size of local model that used by partitioning
        """
        self.leafsize = leafsize

    def train(self, data):
        prepared_data = partitioning.get_local_model_labels(
            data, leaf_size=self.leafsize)
        means, index_data, train_labels = prepared_data

        self.means = means
        self.index_data = index_data
        train_data = index_data[:, :-1]
        self.mlp = model.MLP(
            input_size=train_data.shape[1],
            num_classes=(train_labels[-1, -1]+1)
        )
        self.num_classes = train_labels[-1, -1] + 1

        # print(self.mlp.model)
        loader = DataLoader(TensorDataset(
            train_data, train_labels), shuffle=True, batch_size=100)
        self.mlp.train(loader)

    def query(self, qp, k) -> list:
        output = self.mlp.model(qp)
        output = output.reshape(output.shape[1])
        pred = torch.nonzero(output)[0][0]

        error_rate = self.mlp.error_rate
        error_block_size = int(self.num_classes * error_rate)

        min_block_number = 0 if pred - error_block_size < 0 else (pred - error_block_size)
        max_block_number = pred + error_block_size
        max_block_number = max_block_number if max_block_number < self.num_classes else self.num_classes

        search_points = self.index_data[:, :-1]
        search_points = search_points[
            (search_points[:, -1] >= min_block_number) & 
            (search_points[:, -1] <= max_block_number)
        ]

        norm = torch.norm(search_points - qp, dim=(1))
        topk = torch.topk(norm, k)[1]
        indices = self.index_data[topk, -1]

        return indices
