# Checkout what dropout weights, epoch, generality, etc with CNN-LSTM's
# Installing libraries/frameworks for data manipulation and NN development.
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

# Creating a scalar (O-D tensor).
scalar = torch.tensor(365)
print(scalar.ndim)

# Get scalar back as Python integer.
scalar_int = scalar.item()

# Creating a vector or array (1-D tensor).
array = torch.tensor([6, 23])
print(array.ndim)

# Creating a matrix (2-D tensor).
# Note that with machine learning nomenclature, tensors from 2-D and up are fully capitalized.
MATRIX = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(MATRIX.ndim)

# Get the shape of the tensor (in this case it should print out 3, 3 AKA 3x3).
print(MATRIX.shape)

# Obtain the first row of the matrix.
print(MATRIX[0])

# Creating a 3-D tensor (a series of matrices stacked on each other).
TENSOR = torch.tensor([
    [[1, 2, 4],
     [2, 3, 5]],
    [[-3, 8, 4],
     [2, 32, 15]],
    [[123, 17, 4],
     [24, 13, -9]],
    [[-3, 62, 8],
     [17, 42, 34]]
])
print(TENSOR.ndim)
print(TENSOR.shape) 

# Creating a random tensor. Each parameter in the function below is a dimension of the tensor. So here we've created a 3x4 matrix filled with random numbers.
random_tensor = torch.rand(3, 4)
print(random_tensor)
print(random_tensor.ndim)

# Creating a random tensor with similar shape to an image tensor.
random_tensor_image = torch.rand(224, 224, 3) # height, width, color channels are the arguments of the function call
random_tensor_image2 = torch.rand(3, 224, 224) # color channels can come first too

# Creating a tensor of all zeros and another tensor of all ones.
zeros = torch.zeros(3, 4)
print(zeros)

ones = torch.ones(3, 4)
print(ones)
