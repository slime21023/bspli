import partitioning
import torch
import sys
sys.setrecursionlimit(10000)

data = torch.randint(0, 1000, (10000, 2),  dtype= torch.float)
ids = torch.arange(0, data.shape[0], step=1, dtype= torch.int)
data = torch.hstack((data, ids.reshape(data.shape[0], 1)))

# print(data)
result = partitioning.max_partitioning(data, leaf_size=150)
# for item in result:
#     mean, data = item
#     print(len(data))

local_labels = partitioning.get_local_model_labels(data, leaf_size=150)
print(local_labels)