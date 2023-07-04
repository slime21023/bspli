import sys
sys.path.append("./Bspli/")
import os  
import time
import numpy as np 
import index 
import torch

gist = np.load("dataset/half-gist-960-euclidean.npy")
print(f'gist data shape: {gist.shape}')
print(f'gist dtype: {gist.dtype}')

gist_tensor = torch.from_numpy(gist)
print(f'gist tensor shape: {gist_tensor.shape}')

build_time = 0
start = time.time()
idx = index.Indexing(
    gl_size=40000, 
    ll_size=10000,
    g_epoch_num=3,
    l_epoch_num=10,
    g_hidden_size=5,
    l_hidden_size=5,
    g_block_range=4,
    l_block_range=4,
    random_partitioning=False
)
idx.train(gist_tensor)
build_time += (time.time() - start)

print(f"build time: {build_time}")

if not os.path.exists('./save/gist'):
    os.makedirs('./save/gist')

idx.save('./save/gist')