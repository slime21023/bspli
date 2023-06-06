import sys
sys.path.append("..")
import torch
from utils.Lindex import LIndexing

sys.setrecursionlimit(10000)

data = torch.randint(0, 1000, (10000, 10),  dtype= torch.float)
ids = torch.arange(0, data.shape[0], step=1, dtype= torch.int)
data = torch.hstack((data, ids.reshape(data.shape[0], 1)))


idx = LIndexing(leafsize=150, epoch_num=10)
idx.train(data.to(torch.float))


qp = data[0, :-1]
indices = idx.query(qp, 10)
print(f"pred: {indices}")