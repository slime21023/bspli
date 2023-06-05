import torch
import sys
import index
sys.setrecursionlimit(10000)

data = torch.randint(0, 1000, (25000, 10),  dtype= torch.float)


index = index.Indexing(gl_size=10000, ll_size=150, g_epoch_num= 20, l_epoch_num=10)
index.train(data)

print(f"local model len:{len(index._l_model)}")

qp = data[0]
# print(qp)
pred = index.query(qp, k=100)
pred = pred.to(torch.int)
print(f"pred: {pred}")