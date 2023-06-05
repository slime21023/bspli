import partitioning
import model
import torch
from torch.utils.data import DataLoader, TensorDataset


class LIndexing:
    """
    The local learned indexing model
    To handle the query processing
    """

    def __init__(self, leafsize, epoch_num = 5, hidden_size = 100):
        """
        leafsize: the leaf size of local model that used by partitioning
        """
        self.leafsize = leafsize
        self.epoch_num = epoch_num
        self.hidden_size = hidden_size

    def train(self, data):
        prepared_data = partitioning.get_local_model_labels(
            data, leaf_size=self.leafsize)
        means, index_data, train_labels = prepared_data

        self.means = means
        self.index_data = index_data
        train_data = index_data[:, :-1]
        self.mlp = model.MLP(
            input_size=train_data.shape[1],
            num_classes=(train_labels[-1, -1]+1),
            epoch_num=self.epoch_num,
            hidden_size=self.hidden_size
        )
        self.num_classes = train_labels[-1, -1] + 1

        # print(self.mlp.model)
        loader = DataLoader(TensorDataset(
            train_data, train_labels), shuffle=True, batch_size=100)
        self.mlp.train(loader)

    def query(self, qp, k) -> list:
        qp = qp.reshape(1, qp.shape[0])
        pred = int(self.mlp.model(qp)[0][0])

        # Default search blocks
        error_rate = self.mlp.error_rate
        error_block_size = int(self.num_classes * error_rate)

        min_block_number = 0 if pred - error_block_size < 0 else (pred - error_block_size)
        max_block_number = pred + error_block_size
        max_block_number = max_block_number if max_block_number < self.num_classes else self.num_classes

        def get_search_block(mininal, maximum, k: int):
            search_points = self.index_data[:, :-1]
            search_points = search_points[
                (search_points[:, -1] >= mininal) & 
                (search_points[:, -1] <= maximum)
            ]

            if search_points.shape[0] < k:
                mininal = mininal -1 if mininal > 0 else 0
                maximum = maximum +1 if maximum < self.num_classes else self.num_classes
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
