import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Binary cross-entropy loss function
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Training data for AND gate
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([[0], [0], [0], [1]])

# Initialize randomly  weights and biases
np.random.seed(42)
input_layer_neurons = X_and.shape[1]
hidden_layer_neurons = 2
output_neuron = 1

W1 = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
b1 = np.random.uniform(size=(1, hidden_layer_neurons))
W2 = np.random.uniform(size=(hidden_layer_neurons, output_neuron))
b2 = np.random.uniform(size=(1, output_neuron))
'''
print(W1)
print(b1)
print(W2)
print(b2)
'''
# Set learning rate and number of epochs
alpha = 0.1
epochs = 10000

# To store the loss values
loss_values_and = []

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    Z1 = np.dot(X_and, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # Calculate loss
    loss = binary_cross_entropy(y_and, A2)
    loss_values_and.append(loss)

    # Backpropagation
    dA2 = A2 - y_and
    dZ2 = dA2 * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X_and.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update weights and biases
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W1 -= alpha * dW1
    b1 -= alpha * db1

# Plotting the loss over iterations
plt.figure(figsize=(10, 6))
plt.plot(loss_values_and, label='AND Gate Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss over Iterations for AND Gate')
plt.legend()
plt.show()

# Testing the trained neural network
print("Training complete.")
print("Testing AND Gate Neural Network")

for x, y in zip(X_and, y_and):
    Z1 = np.dot(x, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    prediction = A2 > 0.5
    print(f"Input: {x}, Predicted: {prediction.astype(int)}, Actual: {y}")