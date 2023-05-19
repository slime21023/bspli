import partitioning
import model
from torch.utils.data import DataLoader, TensorDataset

class LIndexing:
    def __init__(self, leafsize):
        """
        leafsize: the leaf size of local model that used by partitioning
        """
        self.leafsize = leafsize

    def train(self, data):
        prepared_data = partitioning.get_local_model_labels(data, leaf_size=self.leafsize)
        means, index_data, train_labels = prepared_data
        
        self.means = means
        self.index_data = index_data
        train_data = index_data[:, :-1]
        print(f"input_size : {train_data.shape[1]}")
        self.mlp = model.MLP(
            input_size=train_data.shape[1],
            num_classes=(train_labels[-1, -1]+1)
        )

        # print(self.mlp.model)
        loader = DataLoader(TensorDataset(train_data, train_labels), shuffle=True, batch_size=100)
        self.mlp.train(loader)

    
    def query(self, qp):
        pass