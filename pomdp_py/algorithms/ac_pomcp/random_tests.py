import torch
import numpy as np

temp = torch.zeros((10))
print(temp)

temp[(10-4):] = torch.tensor([1., 2., 2., 3.])

print(temp)