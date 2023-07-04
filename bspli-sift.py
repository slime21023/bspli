import sys
sys.path.append("./Bspli/")
import os  
import faiss
import time
import numpy as np 
import index 
import torch
import pandas as pd

sift = np.load("dataset/sift-128-euclidean.npy")
print(f'sift data shape: {sift.shape}')

sift_tensor = torch.from_numpy(sift)
print(f'sift tensor shape: {sift_tensor.shape}')

build_time = 0
start = time.time()
idx = index.Indexing(
    gl_size=80000, 
    ll_size=20000,
    g_epoch_num=3,
    l_epoch_num=10,
    g_hidden_size=5,
    l_hidden_size=5,
    g_block_range=4,
    l_block_range=4,
    random_partitioning=False
)
idx.train(sift_tensor)
build_time += (time.time() - start)

print(f"local model len:{len(idx._l_model)}")

def recall(pred, true):
    x = np.isin(pred, true)
    return x.sum() / true.size

# Brute Search
flat = faiss.IndexFlatL2(sift.shape[1])
flat.add(sift)

D, FLAT_I = flat.search(sift[0].reshape(1, sift.shape[1]), k=100) 
print(f'brute query: {FLAT_I}')

qp = torch.from_numpy(sift[0])
# print(qp)
pred = idx.query(qp, k=100)
print(f"recall: {recall(pred, FLAT_I)}")

pred = pred.to(torch.int)
print(f"pred: {pred}")

print(f"total blocks: {idx.get_search_blocks_num()}")
indices = np.random.choice(sift.shape[0], size=1000, replace=False)
##########  Query Benchmark ##############

result = []

def benchmark_knn_query(data, indices, index, g_block_range, l_block_range, k=100):
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
test_range= [ (3, 1), (3, 3), (3, 5), (3, 7), 
             (5, 1), (5, 3), (5, 5), (5, 7), 
             (7, 1), (7, 3), (7, 5), (7, 7), 
             (10, 1), (10, 3), (10, 5), (10, 7),
             (13, 1), (13, 3), (13, 5), (13, 7),
             (15, 1), (15, 3), (15, 5), (15, 7),
             (17, 1), (17, 3), (17, 5), (17, 7),
             (20, 1), (20, 3), (20, 5), (20, 7),
             (23, 1), (23, 3), (23, 5), (23, 7),
             ]

for block_range in test_range:
    benchmark_knn_query(sift, indices, idx, block_range[0], block_range[1], k=10)

print(result)
benchmark_df = pd.DataFrame(result, columns=['query_time', 'recall', 'g_range', 'l_range'])
benchmark_df.to_csv("./results/sift-benchmark.csv",index=False)

##########  Query with threshold Benchmark ##############

result = []

def benchmark_knn_query_with_threshold(data, indices, index, g_block_range, threshold=0.5, k=100):
    query_time = 0
    cur_recall = 0

    # query
    for i in indices:
        q = torch.from_numpy(data[i])
        start = time.time()
        qk = index.query_with_threshold(q,  g_block_range=g_block_range, threshold=threshold, k=100)
        query_time += (time.time() - start)
        D, FLAT_I = flat.search(data[i].reshape(1, data.shape[1]), k=k) 
        cur_recall += recall(qk, FLAT_I)
    
    item = (query_time/1000, cur_recall/1000, g_block_range, threshold)
    print(f"result: {item}")
    result.append(item)


test_range = [ (3, 0.5), (3, 0.1), (3, 0.07), (3, 0.05),
               (5, 0.5), (5, 0.1), (5, 0.07), (5, 0.05),
               (7, 0.5), (7, 0.1), (7, 0.07), (7, 0.05), 
               (10, 0.5), (10, 0.1), (10, 0.07), (10, 0.05),
               (13, 0.5), (13, 0.1), (13, 0.07), (13, 0.05),
               (15, 0.5), (15, 0.1), (15, 0.07), (15, 0.05),
               (17, 0.5), (17, 0.1), (17, 0.07), (17, 0.05),
               (20, 0.5), (20, 0.1), (20, 0.07), (20, 0.05),
               (23, 0.5), (23, 0.1), (23, 0.07), (23, 0.05),
             ]

for block_range in test_range:
    benchmark_knn_query_with_threshold(sift, indices, idx, block_range[0], block_range[1], k=10)


print(result)
benchmark_threshold_df = pd.DataFrame(result, columns=['query_time', 'recall', 'g_range', 'threshold'])
benchmark_threshold_df.to_csv("./results/sift-benchmark-threshold.csv",index=False)
