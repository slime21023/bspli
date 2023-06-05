import model
import partitioning
import torch
from torch.utils.data import DataLoader, TensorDataset


class GIndexing:
    """
    The global learned indexing model
    To predict the query point in which local learned indexing model
    """

    def __init__(self, leafsize=2, epoch_num = 5, hidden_size = 100):
        self.leafsize = leafsize
        self.epoch_num = epoch_num
        self.hidden_size = hidden_size

    def train(self, model_list: list):
        """
        model_list: the list of the local models
        """
        prepared_data = partitioning.get_global_model_labels(model_list)
        print(f'global index train smaple count: {prepared_data.shape[0]}')
        self.index_data = prepared_data
        train_data = prepared_data[:, :-1]
        train_labels = prepared_data[:, -1]

        self.mlp = model.MLP(
            input_size= train_data.shape[1],
            num_classes=int(train_labels[-1] + 1),
            epoch_num=self.epoch_num,
            hidden_size=self.hidden_size
        )

        self.num_classes = (train_labels[-1] + 1)

        loader = DataLoader(TensorDataset(
            train_data, train_labels
        ), shuffle=True, batch_size=5)
        self.mlp.train(loader)

    def query(self, qp):
        qp = qp.reshape(1, qp.shape[0])
        output = self.mlp.model(qp)
        output = output.reshape(output.shape[1])    
        pred = int(self.mlp.model(qp)[0][0])
        return pred
        