# Aditya Bhati - CSE 5523 HW3
import numpy as np

# Sample data points from the figure in the HW3 pdf (originally in 2D).
X_plain = np.array([
    [-0.5, 0.5],
    [0.5, -0.5],
    [-1.0, 4.0],
    [4.0, -2.0],
    [-6.0, 3.0],
    [-1.0, -5.0],
    [1.0, 5.0],
], dtype=float)

# Sample y labels for each of the data points (blue = +1, red = -1)
y = np.array([+1, -1, +1, -1, +1, -1, +1], dtype=float)

# Defining the extended feature space by appending a bias term (1) to each data point.
ones = np.ones((X_plain.shape[0], 1))

# This matrix consists of extended feature vectors for each of the data points as rows, where the last column corresponds to the bias term.
X = np.hstack((X_plain, ones))

def perceptron(X, y):
    # Initializing the parameter vector w to zeros, creating a check to see if the algorithm has converged or not, and initializing the number of iterations taken.
    w = np.array([0.0, 0.0, 0.0])
    converged = False
    n_iterations = 0

    while not converged:
        # Assuming convergence until we find a misclassified point, then it will be set to False.
        converged = True

        for i in range(len(X)):
            # The condition y[i] * np.dot(w, X[i]) <= 0 checks if the data point X[i] is misclassified by the current parameter vector w.
            if y[i] * np.dot(w, X[i]) <= 0:
                # If so, w is updating as follows, converged is set to False, and the number of iterations is incremented by 1.
                w = w + y[i] * X[i]
                converged = False
                n_iterations += 1

                # Printing the intermediate update to w after each misclassification.
                print(f"Iteration {n_iterations}: Updated w to {w}")

    return w, n_iterations

# Running the perceptron function defined above using the extended feature space and its corresponding labels.
w_final, total_iterations = perceptron(X, y)
print(f"Final parameter vector w: {w_final}")
print(f"Total number of iterations until convergence: {total_iterations}")