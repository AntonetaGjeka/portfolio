import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data = pd.read_csv("predictive_maintenance.csv")

# Step 2.1 & 2.2: Drop the unwanted columns
data.drop(["UDI", "Product ID", "Type","Failure Type"], axis=1, inplace=True)

# Step 2.3: Standardize the input features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop("Target", axis=1)) # Scaling all features except the target
target = data['Target'].astype(int)

# Step 3: Dataset splitting, 80% train, 20% test
train_size = int(0.8 * len(data))
X_train = scaled_features[:train_size]
y_train = target[:train_size]
X_test = scaled_features[train_size:]
y_test = target[train_size:]

# Step 4: Implement logistic regression:
def sigmoid(h):
    return 1 / (1 + np.exp(-h))

def compute_loss(y, h):
    return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

def gradient_descent(X_train, y_train, X_test, y_test, learning_rate, iterations):
    loss_history_train = []
    loss_history_test = []
    theta_history = []
    theta_0 = 1
    theta = np.ones(X_train.shape[1])
    for i in range(iterations):
        #Train calculations
        l_train = np.dot(X_train, theta) + theta_0
        h_train = sigmoid(l_train)
        loss_train = compute_loss(y_train, h_train)
        loss_history_train.append(loss_train)

        #Test calculations
        l_test = np.dot(X_test, theta) + theta_0
        h_test = sigmoid(l_test)
        loss_test = compute_loss(y_test, h_test)
        loss_history_test.append(loss_test)

        #Calculate Gradient
        gradient = np.dot(X_train.T, (h_train - y_train)) / len(y_train)
        gradient_0 = np.mean(h_train-y_train)
        #Update the theta values
        theta -= learning_rate * gradient
        theta_0 -= learning_rate * gradient_0
        theta_history.append(theta.copy())
    return theta, theta_0, loss_history_train, loss_history_test, theta_history

lr = 0.01  # Adjust the learning rate
iterations = 1500

theta, theta_0, loss_history_train, loss_history_test, theta_history = gradient_descent(X_train, y_train, X_test, y_test, lr, iterations)
# Calculate accuracy
y_pred_train = sigmoid(np.dot(X_train, theta) + theta_0) >= 0.5
train_accuracy = np.mean(y_pred_train == y_train)
y_pred_test = sigmoid(np.dot(X_test, theta) + theta_0) >= 0.5
test_accuracy = np.mean(y_pred_test == y_test)
print(f'Train Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')
print('MSE(train) = ', loss_history_train[-1])
print('MSE(test) = ', loss_history_test[-1])

#Plot the loss & convergence of parameters:
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(loss_history_train, label='Training Loss', color='blue')
plt.plot(loss_history_test, label='Testing Loss', linestyle='--', color='red')
plt.title("Loss over iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()

theta_history = np.array(theta_history)
plt.subplot(1, 2, 2)
for j in range(theta_history.shape[1]):
    plt.plot([row[j] for row in theta_history], label=f'Theta {j}')
plt.title('Convergence of Thetas')
plt.xlabel('Iterations')
plt.ylabel('Theta Values')
plt.legend()
plt.tight_layout()
plt.show()

