import torch
import sys
import index
sys.setrecursionlimit(10000)

data = torch.randint(0, 1000, (25000, 10),  dtype= torch.float)


index = index.Indexing(gl_size=10000, ll_size=150)
index.train(data)

print(f"local model len:{len(index._l_model)}")

qp = data[50]
# print(qp)
pred = index.query(qp, k=50)
print(f"pred: {pred}")