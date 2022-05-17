import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
if __name__ == "__main__":
    tensors = []
    for i in range(4):
        tensors.append(torch.randn(30*1024*1024*1024 // 4, dtype=torch.float32, device=torch.device("cuda:{}".format(i))))
    
    while True:
        for i in range(4):
            tensors[i] *= 1
            tensors[i] += 0
            tensors[i] /= 1