import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# Output dataset
y = np.array([[0],
              [1],
              [1],
              [0]])

# Seed the random number generator for consistent results
np.random.seed(1)

# Initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 4)) - 1  # Weights between input and hidden layer
syn1 = 2 * np.random.random((4, 1)) - 1  # Weights between hidden and output layer

# Training loop
for j in range(60000):
    # Forward propagation
    l0 = X  # Input layer
    l1 = sigmoid(np.dot(l0, syn0))  # Hidden layer output
    l2 = sigmoid(np.dot(l1, syn1))  # Output layer output

    # Error calculation
    l2_error = y - l2

    # Print error every 10000 iterations
    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))

    # Backpropagation
    l2_delta = l2_error * sigmoid_derivative(l2)  # Output layer error delta
    l1_error = l2_delta.dot(syn1.T)  # Hidden layer error
    l1_delta = l1_error * sigmoid_derivative(l1)  # Hidden layer error delta

    # Update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)