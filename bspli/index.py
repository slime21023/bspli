from utils.Partitioning import random_partitioning, max_partitioning
from utils.Gindex import GIndexing
from utils.Lindex import LIndexing

import torch
import sys
sys.setrecursionlimit(100000)


class Indexing:
    def __init__(self, gl_size, ll_size, 
                 g_epoch_num=10, 
                 l_epoch_num=5, 
                 g_hidden_size=100, 
                 l_hidden_size=100,
                 g_block_range=1,
                 l_block_range=5,
                 random_partitioning=False):
        """
        gl_size: the leaf size of global model that used by partitioning
        ll_size: the leaf size of local model that used by partitioning
        """
        self._gl_size = gl_size
        self._ll_size = ll_size
        self.g_epoch_num = g_epoch_num
        self.l_epoch_num = l_epoch_num
        self.g_hidden_size= g_hidden_size
        self.l_hidden_size = l_hidden_size
        self.g_block_range = g_block_range
        self.l_block_range = l_block_range
        self.random_partitioning = random_partitioning
        self._g_model = None
        self._l_model = []

    def train(self, data):
        ids = torch.arange(0, data.shape[0], step=1, dtype= torch.int)
        prepared_data = torch.hstack((data, ids.reshape(data.shape[0], 1)))
        
        # To create the local index model
        def generate_local_model(data):
            print("training local model")
            lin = LIndexing(
                self._ll_size, 
                epoch_num=self.l_epoch_num, 
                hidden_size=self.l_hidden_size,
                block_range=self.l_block_range
            )

            lin.train(data)
            return lin

        # Partitioning for the local index model
        if self.random_partitioning == True:
            first_stage_partitioning = random_partitioning(prepared_data, leaf_size=self._gl_size)
        else:
            first_stage_partitioning = max_partitioning(prepared_data, leaf_size=self._gl_size)
            first_stage_partitioning = list(map(lambda item: item[1], first_stage_partitioning))

        for block in first_stage_partitioning:
            print(block.shape)
        print("first stage partitioning finish")
        print(f'partitioning blocks : {len(first_stage_partitioning)}')

        # Train the local index models
        self._l_model = list(map(generate_local_model,  first_stage_partitioning))


        # To Handle the class imbalance for means (global model)
        def resampling(means: list) -> list:
            max_num = 0
            for m in means:
                max_num = m.shape[0] if m.shape[0] > max_num else max_num

            for idx, m in enumerate(means):
                indices = torch.randint(m.shape[0], size=(max-m.shape[0],))
                means[idx] = torch.vstack(m, m[indices]) 
            
            return means


        # Train the global index model
        means = list(map(lambda i: (i.means,), self._l_model))
        self._g_model = GIndexing(
            leafsize=self._gl_size, 
            epoch_num=self.g_epoch_num,
            block_range=self.g_block_range,
            hidden_size=self.g_hidden_size
        )
        print("trainging global model")
        self._g_model.train(model_list=means)
        print("finish")

    def query(self, qp, k):
        # predict the query point in which local model
        pred = self._g_model.query(qp)

        # get the topk result indices
        indices = self._l_model[pred].query(qp, k)

        return indices