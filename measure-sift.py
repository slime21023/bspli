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
print(f'sift dtype: {sift.dtype}')

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

print(f"build time: {build_time}")

########### measure predict time ###########

result = []

def measure_predict_time(data, index, g_block_range, l_block_range, size=1000):
    indices = np.random.choice(data.shape[0], size, replace=False)
    predict_time = 0

    for i in indices:
        q = torch.from_numpy(data[i])
        predict_time += index.measure_time(q, g_block_range=g_block_range, l_block_range=l_block_range)

    item = (predict_time/1000, g_block_range, l_block_range)
    print(f"result: {item}")
    result.append(item)

test_range = [
    (3, 1), (3, 3), (3, 5), (3, 7), 
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
    measure_predict_time(sift, idx, block_range[0], block_range[1], size=1000)

print(result)
measure_df = pd.DataFrame(result, columns=['predict_time', 'g_range', 'l_range'])
measure_df["build_time"] = build_time

measure_df.to_csv("./results/measure-time-sift.csv",index=False)

##########  Query with threshold Benchmark ##############

result = []

def measure_predict_time_with_threshold(data, index, g_block_range, threshold=0.5, size=1000):
    indices = np.random.choice(data.shape[0], size, replace=False)
    predict_time = 0

    for i in indices:
        q = torch.from_numpy(data[i])
        predict_time += index.measure_time_with_threshold(q, g_block_range=g_block_range, threshold=threshold)

    item = (predict_time/1000, g_block_range, threshold)
    print(f"result: {item}")
    result.append(item)


test_range = [ 
    (3, 0.5), (3, 0.1), (3, 0.07), (3, 0.05),
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
    measure_predict_time_with_threshold(sift, idx, block_range[0], block_range[1], size=1000)

print(result)
measure_threshold_df = pd.DataFrame(result, columns=['predict_time', 'g_range', 'threshold'])
measure_threshold_df["build_time"] = build_time

measure_threshold_df.to_csv("./results/measure-time-threshold-sift.csv",index=False)