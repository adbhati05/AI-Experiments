import tensorflow as tf
import numpy as np

# Creating a scalar (0-D tensor)
zeroDimen = tf.constant(4)
print(zeroDimen)

# Creating an array (1-D tensor).
oneDimen = tf.constant([1, 1, 2, 3, 5, 8, 13])
print(oneDimen)

# Creating a matrix (2-D tensor).
twoDimen = tf.constant([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(twoDimen)

# Creating a 3-D tensor. Note that between each dimension, commas need to be added. So for instance, in arrays, add commas between the scalars. In matrices, add commas between the arrays. And in 3-D tensors, add commas between the matrices. 
threeDimen = tf.constant([
    [[1, 2, 4],
     [2, 3, 5]],
    [[-3, 8, 4],
     [2, 32, 15]],
    [[123, 17, 4],
     [24, 13, -9]],
    [[-3, 62, 8],
     [17, 42, 34]]
])
print(threeDimen)

# Creating a 4-D tensor. Functions from tensorflow like ndim and shape return the size and dimension of the tensor. 
# Also, you can add arguments to tf.constant such as the data type you want to specify for the tensor (dtype=tf.float32 for instance). Realize how the size of the float is specified, that's because you can adjust how much memory you want to allocate for this data structure (that is if you know the data well enough to know what size is suitable).
# Refer to TF's documentation for all the data types available.
# Another function from tensorflow is tf.cast, which allows you to cast a certain tensor's data to a certain data type. Look below for example.
fourDimen = tf.constant([
    [[[1, 2, 4],
      [2, 3, 5]],
     [[-3, 8, 4],
      [2, 32, 15]],
     [[123, 17, 4],
      [24, 13, -9]],
     [[-3, 62, 8],
      [17, 42, 34]]],
    [[[3, 13, 24],
      [-14, 8, 5]],
     [[-32, 28, 4],
      [2, 332, 125]],
     [[-23, 17, 4],
     [224, 143, -9]],
     [[-3, 42, 8],
      [17, 5, -40]]]
], dtype=tf.float32)

# Also note that data can come in many forms, strings, numbers, decimals, etc. With tf, you're not limited to just floating points, you're also able to use strings. 
# If you want to convert, say an array, into a tensor look below (used NumPy to create an array).
fourDimenCasted = tf.cast(fourDimen, dtype=tf.int16)
print(fourDimen)
print(fourDimenCasted)

arr = np.array([1, 2, 3, 5])
print(arr)
converted = tf.convert_to_tensor(arr)
print(converted)

# You can create an identity matrix as follows:
# If you want to create a 3-D tensor, adjust the value of batch_shape.
identityMat = tf.eye(num_rows=3, num_columns=None, batch_shape=None, dtype=tf.dtypes.int16, name=None)
print(identityMat)

# Here, batch_shape being set to [3,] ensures that three 5x5 matrices are stacked on each other.
batchShapeEx = tf.eye(num_rows=5, num_columns=None, batch_shape=[3,], dtype=tf.dtypes.int16, name=None)
print(batchShapeEx)

# If you want to set all the values in a 3x4 matrix to be 5, here's the code:
filled = tf.fill([3, 4], 5, name=None)
print(filled)

# Using tf.ones_like to convert the matrix specified above to be one filled with ones. 
print(tf.ones_like(filled))

# Using tf.zeros_like to do the same as above.
print(tf.zeros_like(filled))

# Creating a tensor with random values (as you can see you can adjust the mean and standard deviation of the values in the set):
# Note that here we're using a normal distribution of values (refer to Statistics notes, or just look online for what the different types of probability distributions are).
# A normally distributed set of values is defined by a symmetrical bell curve around the mean (so this bell curve should be built around 15.7 and should generally deviate by 5 from each side). 
# With normal distributions, 68% of values are within one standard deviation of the mean, 95% within two standard deviations, and 99.7% within 3 standard deviations.
randomNormal= tf.random.normal([3, 3, 2], mean=15, stddev=5, dtype=tf.dtypes.float16, seed=None, name=None)
print(randomNormal)

# Here's another example of a randomly curated tensor with values that follow a uniform distribution.
randomUniform = tf.random.uniform([3, 3, 2], minval=-25, maxval=25, dtype=tf.dtypes.float16, seed=None, name=None)
print(randomUniform)