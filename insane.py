import torch
import numpy as np
import time


arr1 = np.random.rand(3840,2160,3)
arr2 = np.random.rand(3840,2160,3)

dev = torch.device("cuda")

start = time.time()

a = torch.tensor(arr1, device=dev)
b = torch.tensor(arr2, device=dev)

end = time.time()
print(end - start)

start = time.time()

c = a + b

print(c.cpu())

end = time.time()
print(end - start)


