import partitioning
import gindex
import lindex
import torch
import sys
sys.setrecursionlimit(100000)


class Indexing:
    def __init__(self, gl_size, ll_size):
        """
        gl_size: the leaf size of global model that used by partitioning
        ll_size: the leaf size of local model that used by partitioning
        """
        self._gl_size = gl_size
        self._ll_size = ll_size
        self._g_model = None
        self._l_model = []

    def train(self, data):
        ids = torch.arange(0, data.shape[0], step=1, dtype= torch.int)
        prepared_data = torch.hstack((data, ids.reshape(data.shape[0], 1)))
        
        # To create the local index model
        def generate_local_model(data):
            print("training local model")
            lin = lindex.LIndexing(self._ll_size)
            lin.train(data)
            return lin

        # Random Partitioning for the local index model
        first_stage_partitioning = partitioning.random_partitioning(prepared_data, leaf_size=self._gl_size)
        for block in first_stage_partitioning:
            print(block.shape)
        print("Ramdom partitioning finish")

        # Train the local index models
        self._l_model = list(map(generate_local_model,  first_stage_partitioning))

        # Train the global index model
        means = list(map(lambda i: (i.means,), self._l_model))
        self._g_model = gindex.GIndexing(leafsize=self._gl_size)
        print("trainging global model")
        self._g_model.train(model_list=means)
        print("finish")

    def query(self, qp, k):
        # qp = qp.reshape(1, qp.shape[0])
        # predict the query point in which local model
        pred = self._g_model.query(qp)
        print(f"predicted local model: {pred}")

        # get the topk result indices
        indices = self._l_model[pred].query(qp, k)

        return indices