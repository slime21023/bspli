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
    (10, 1), (10, 3), (10, 5), (10, 7)
]

for block_range in test_range:
    measure_predict_time(mnist, idx, block_range[0], block_range[1], size=1000)

print(result)
measure_df = pd.DataFrame(result, columns=['predict_time', 'g_range', 'l_range'])
measure_df["build_time"] = build_time

measure_df.to_csv("./results/measure-time-mnist.csv",index=False)

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
    (3, 0.5), (3, 0.1),  (3, 0.07), (3, 0.05), 
    (5, 0.5), (5, 0.1),  (5, 0.07), (5, 0.05), 
    (7, 0.5), (7, 0.1),  (7, 0.07), (7, 0.05), 
    (10, 0.5), (10, 0.1),  (10, 0.07), (10, 0.05)
]

for block_range in test_range:
    measure_predict_time_with_threshold(mnist, idx, block_range[0], block_range[1], size=1000)

print(result)
measure_threshold_df = pd.DataFrame(result, columns=['predict_time', 'g_range', 'threshold'])
measure_threshold_df["build_time"] = build_time

measure_threshold_df.to_csv("./results/measure-time-threshold-mnist.csv",index=False)