import sys
sys.path.append("./Bspli/")
import os  
import faiss
import time
import numpy as np 
import index 
import torch
import pandas as pd

mnist = np.load("dataset/mnist-784-euclidean.npy")
print(f'mnist data shape: {mnist.shape}')
print(f'mnist dtype: {mnist.dtype}')

mnist_tensor = torch.from_numpy(mnist)
print(f'mnist tensor shape: {mnist_tensor.shape}')

build_time = 0
start = time.time()
idx = index.Indexing(
    gl_size=10000, 
    ll_size=2000,
    g_epoch_num=3,
    l_epoch_num=10,
    g_hidden_size=5,
    l_hidden_size=5,
    g_block_range=4,
    l_block_range=4,
    random_partitioning=False
)
idx.train(mnist_tensor)
build_time += (time.time() - start)

print(f"local model len:{len(idx._l_model)}")

def recall(pred, true):
    x = np.isin(pred, true)
    return x.sum() / true.size

# Brute Search
flat = faiss.IndexFlatL2(mnist.shape[1])
flat.add(mnist)

D, FLAT_I = flat.search(mnist[0].reshape(1, mnist.shape[1]), k=100) 
print(f'brute query: {FLAT_I}')

qp = torch.from_numpy(mnist[0])
# print(qp)
pred = idx.query(qp, k=100)
print(f"recall: {recall(pred, FLAT_I)}")

pred = pred.to(torch.int)
print(f"pred: {pred}")

print(f"total blocks: {idx.get_search_blocks_num()}")


result = []

def benchmark_knn_query(data, index, g_block_range, l_block_range, size=1000, k=100):
    indices = np.random.choice(data.shape[0], size, replace=False)
    query_time = 0
    cur_recall = 0

    # query
    for i in indices:
        q = torch.from_numpy(data[i])
        start = time.time()
        qk = index.query(q,  g_block_range=g_block_range, l_block_range=l_block_range, k=100)
        query_time += (time.time() - start)
        D, FLAT_I = flat.search(data[i].reshape(1, data.shape[1]), k=k) 
        cur_recall += recall(qk, FLAT_I)
    
    item = (query_time/1000, cur_recall/1000, g_block_range, l_block_range)
    print(f"result: {item}")
    result.append(item)

# result item: (query_time, recall, g_block_range, l_block_range)
test_range= [ (3, 1), (3, 5), (3, 7), (5, 1), (5, 5), (5, 7), (7, 1), (7, 5), (7, 7), (10, 1), (10, 5), (10, 7)]

for block_range in test_range:
    benchmark_knn_query(mnist, idx, block_range[0], block_range[1], size=1000, k=10)

print(result)

