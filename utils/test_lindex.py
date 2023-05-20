import torch
import lindex
import sys
sys.setrecursionlimit(10000)

data = torch.randint(0, 1000, (10000, 2),  dtype= torch.float)
ids = torch.arange(0, data.shape[0], step=1, dtype= torch.int)
data = torch.hstack((data, ids.reshape(data.shape[0], 1)))


idx = lindex.LIndexing(leafsize=150)
idx.train(data.to(torch.float32))


qp = data[0, :-1]
qp = qp.reshape(1, qp.shape[0])
indices = idx.query(qp, 10)
print(f"pred: {indices}")