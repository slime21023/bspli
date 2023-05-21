import model
import partitioning
import torch
from torch.utils.data import DataLoader, TensorDataset


class GIndexing:
    """
    The global learned indexing model
    To predict the query point in which local learned indexing model
    """

    def __init__(self, leafsize=2):
        self.leafsize = leafsize

    def train(self, model_list: list):
        """
        model_list: the list of the local models
        """
        prepared_data = partitioning.get_global_model_labels(model_list)
        self.index_data = prepared_data
        train_data = prepared_data[:, :-1]
        train_labels = prepared_data[:, -1]

        self.mlp = model.MLP(
            input_size= train_data.shape[1],
            num_classes=int(train_labels[-1] + 1)
        )

        self.num_classes = (train_labels[-1] + 1)

        loader = DataLoader(TensorDataset(
            train_data, train_labels
        ), shuffle=True, batch_size=4)
        self.mlp.train(loader)

    def query(self, qp):
        qp = qp.reshape(1, qp.shape[0])
        output = self.mlp.model(qp)
        output = output.reshape(output.shape[1])    
        pred = torch.nonzero(output)[0][0]
        return pred
        