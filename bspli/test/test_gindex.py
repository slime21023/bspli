import sys
sys.path.append("..")
import torch

from utils.Partitioning import random_partitioning, get_global_model_labels, get_local_model_labels
from utils.Gindex import GIndexing
sys.setrecursionlimit(10000)

data = torch.randint(0, 1000, (10000, 10),  dtype= torch.float)
ids = torch.arange(0, data.shape[0], step=1, dtype= torch.int)
data = torch.hstack((data, ids.reshape(data.shape[0], 1)))


pdata = random_partitioning(data, leaf_size=2000)

def generate_model_list(item):
    return get_local_model_labels(item, leaf_size=150)

pdata = list(map(generate_model_list, pdata))

means, data, ids = pdata[0]

labels = get_global_model_labels(pdata)
# print(labels)

index = GIndexing()
index.train(pdata)

qp = means[0]
print(qp.shape)


pred = index.query(qp)
print(f"pred: {pred}")
