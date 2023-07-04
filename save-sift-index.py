import sys
sys.path.append("./Bspli/")
import os  
import time
import numpy as np 
import index 
import torch

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

if not os.path.exists('./save/sift'):
    os.makedirs('./save/sift')

idx.save('./save/sift')