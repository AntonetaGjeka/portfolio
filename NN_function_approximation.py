import numpy as np
import matplotlib.pyplot as plt

# Define the function
def target_function(x):
    return 3 * x**5 + 1.5 * x**4 + 2 * x**3 + 7 * x + 0.5

# ReLU activation function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Mean Squared Error loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Generate data points
np.random.seed(42)
x_data = np.linspace(-1, 1, 100).reshape(-1, 1)
y_data = target_function(x_data)

# Initialize weights and biases
input_neurons = 1
hidden_neurons = 8
output_neurons = 1

W1 = np.random.randn(input_neurons, hidden_neurons)
b1 = np.random.randn(1, hidden_neurons)
W2 = np.random.randn(hidden_neurons, hidden_neurons)
b2 = np.random.randn(1, hidden_neurons)
W3 = np.random.randn(hidden_neurons, output_neurons)
b3 = np.random.randn(1, output_neurons)

# Set learning rate and number of epochs
alpha = 0.01
epochs = 1000

# To store the loss values
loss_values = []

# Training the neural network
for epoch in range(epochs):
    # Forward propagation
    Z1 = np.dot(x_data, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    y_pred = Z3  # Linear activation for the output layer

    # Calculate loss
    loss = mse_loss(y_data, y_pred)
    loss_values.append(loss)

    # Backpropagation
    dZ3 = y_pred - y_data
    dW3 = np.dot(A2.T, dZ3) / x_data.shape[0]
    db3 = np.sum(dZ3, axis=0, keepdims=True) / x_data.shape[0]

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / x_data.shape[0]
    db2 = np.sum(dZ2, axis=0, keepdims=True) / x_data.shape[0]

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(x_data.T, dZ1) / x_data.shape[0]
    db1 = np.sum(dZ1, axis=0, keepdims=True) / x_data.shape[0]

    # Update weights and biases
    W3 -= alpha * dW3
    b3 -= alpha * db3
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W1 -= alpha * dW1
    b1 -= alpha * db1

# Plotting the loss over iterations
plt.figure(figsize=(10, 6))
plt.plot(loss_values, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.legend()
plt.show()

# Compare the predicted values to the actual function values
y_pred = np.dot(relu(np.dot(relu(np.dot(x_data, W1) + b1), W2) + b2), W3) + b3

plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, label='Actual', color='blue')
plt.plot(x_data, y_pred, label='Predicted', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Neural Network Prediction vs Actual Function')
plt.legend()
plt.show()

# Print final predictions for verification
'''
print("Final predictions vs actual values:")
for i in range(len(x_data)):
    print(f"Input: {x_data[i][0]:.2f}, Predicted: {y_pred[i][0]:.2f}, Actual: {y_data[i][0]:.2f}")
'''