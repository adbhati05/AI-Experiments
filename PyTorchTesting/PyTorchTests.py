# Checkout what dropout weights, generality, etc with CNN-LSTM's
# Also check out the PyTorch documentation for the latest updates and changes as well as what each function does
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

# Creating a tensor with a range of values.
torch_range = torch.arange(1, 11, 1) # Start (starting value), end (ending value), step (how far apart each value is) are the args.
print(torch_range)

# Creating tensor-like objects. Here we create a tensor of the same shape as the torch_range but filled with zeros.
ten_zeros = torch.zeros_like(torch_range)
print(ten_zeros)

# Creating a tensors with specific data types (the numbers signify the number of bits used to represent the number, which limnits the range of values that can be used).
float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
print(float_tensor)

int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
print(int_tensor)

# Parameters of torch.tensor and what they do (tensor attributes).
test = torch.tensor(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]],
    dtype=torch.float32, # Data type of the tensor.
    device="cpu", # What we're using to process the tensor. Options are CPU or GPU (which is recommended for ML). Silicon Macbooks can only use CPU so figure out how to use GPU on perhaps a remote device, etc.
    requires_grad=True # Whether or not we want to track gradients for this tensor. This is important for backpropagation in neural networks.
)
print(test.dtype)
print(test.shape)
print(test.device)

# Tensor manipulation, addition, multiplication, subtraction, division.
addition = torch.tensor([1, 2, 3]) + torch.tensor([4, 5, 6])
print(addition)

subtraction = torch.tensor([1, 2, 3]) - torch.tensor([4, 5, 6])
print(subtraction)

multiplication = torch.tensor([1, 2, 3]) * torch.tensor([4, 5, 6])
print(multiplication)

division = torch.tensor([4, 6, 8]) / torch.tensor([2, 3, 4])
print(division)

# Tensor manipulation, matrix multiplication.
matrix_multiplication = torch.matmul(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]]))
print(matrix_multiplication)

# Finding the min, max, mean, median, and standard deviation of a tensor. Make sure to use the correct data type for the tensor (float32 or float64) to avoid errors.
test_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
print(f"Tensor minimum: {test_tensor.min()}")
print(f"Tensor maximum: {test_tensor.max()}")
print(f"Tensor mean: {test_tensor.mean()}")
print(f"Tensor median: {test_tensor.median()}")
print(f"Tensor std deviation: {test_tensor.std()}")
print(f"Tensor sum: {test_tensor.sum()}")

# Finding the index of the minimum and maximum values in a tensor.
print(f"Index of minimum value: {test_tensor.argmin()}")
print(f"Index of maximum value: {test_tensor.argmax()}")

# Tensor reshaping, stacking, splitting, concatenating, squeezing, unsqueezing, permuting, and transposing.
reshaped_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]]).reshape(3, 2) # Keep a track of the original tensor shape and the new shape to avoid errors (dimensions must have relation).
print(reshaped_tensor)

transposed_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]]).T
print(transposed_tensor)

concatenated_tensor = torch.cat((torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])), dim=0) # Concatenating the two tensors along the first dimension (0 in this case)
print(concatenated_tensor)

stacked_tensor = torch.stack((torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]])), dim=0) # Stacking the two tensors along a new dimension (0 in this case)
print(stacked_tensor)

split_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]).split(2) # Splitting the 4x3 matrix into two 2x3 matrices (essentially splitting the tensor into 2 chunks).
print(split_tensor)

squeezed_tensor = torch.tensor([[1], [2], [3]]).squeeze() # Removing the first dimension (0 in this case) from the tensor.
print(squeezed_tensor)

unsqueezed_tensor = torch.tensor([1, 2, 3]).unsqueeze(0) # Adding a new dimension (0 in this case) to the tensor.
print(unsqueezed_tensor)

permuted_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]]).permute(1, 0) # Permuting the dimensions of the tensor (swapping the first and second dimensions in this case).
print(permuted_tensor)

# Changing the view of a tensor. Bear in mind that this does not create a new tensor, but rather changes the view of the original tensor (a view of a tensor shares the same memory as the original tensor).
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = x.view(2, 5) # Changing the view of the tensor to a 2x5 matrix.
print(x)
print(y)

y[0, 0] = 100 # Changing the first element of the first row of the new tensor.
print(x) # Realize how the first element of the original tensor is also changed since they share the same memory.
print(y)