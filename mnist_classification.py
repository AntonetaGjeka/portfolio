import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')
X, y = mnist['data'].to_numpy(), mnist['target'].astype(np.int32)

# Normalize the input data
X = X / 255.0

# Encode the labels
y = np.eye(10)[y]

# Split the data into training and test sets
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Ensure X_test is a NumPy array
X_test = np.array(X_test)

# Define the network architecture
input_size = 784  # 28x28 images flattened
hidden_size = 128
output_size = 10

# Improved weight initialization using He initialization
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
b2 = np.zeros((1, output_size))


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return Z > 0


def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def compute_loss(A2, Y):
    m = Y.shape[0]
    log_probs = -np.log(A2[range(m), np.argmax(Y, axis=1)])
    loss = np.sum(log_probs) / m
    return loss


def backward_propagation(X, Y, Z1, A1, Z2, A2):
    m = X.shape[0]

    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


def update_parameters(dW1, db1, dW2, db2, learning_rate):
    global W1, b1, W2, b2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2


def train(X, Y, iterations, learning_rate):
    loss_history = []
    for i in range(iterations):
        # Forward propagation
        Z1, A1, Z2, A2 = forward_propagation(X)

        # Compute the loss
        loss = compute_loss(A2, Y)
        loss_history.append(loss)

        # Backward propagation
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2)

        # Update parameters
        update_parameters(dW1, db1, dW2, db2, learning_rate)

        # Print the loss for the epoch
        if i % 50 == 0:
            print(f"Iteration {i}, Loss: {loss}")

    return loss_history


# Train the neural network
loss_history = train(X_train, y_train, iterations=300, learning_rate=0.1)

# Plot the training loss
plt.plot(loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()


def predict(X):
    _, _, _, A2 = forward_propagation(X)
    return np.argmax(A2, axis=1)


# Evaluate on the test set
predictions = predict(X_test)
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {accuracy*100}%")


# Display a few test images with predicted labels
def display_images(images, labels, predictions, n=3):
    plt.figure(figsize=(10, 4))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {predictions[i]}, True: {labels[i]}")
        plt.axis('off')
    plt.show()


# Display images with predictions
display_images(X_test, np.argmax(y_test, axis=1), predictions)
