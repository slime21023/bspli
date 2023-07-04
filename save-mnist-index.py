import sys
sys.path.append("./Bspli/")
import os  
import time
import numpy as np 
import index 
import torch

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

if not os.path.exists('./save/mnist'):
    os.makedirs('./save/mnist')

idx.save('./save/mnist')