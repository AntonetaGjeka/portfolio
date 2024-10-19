import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data = pd.read_csv("Real_estate.csv")

# Step 2: Data labeling
mean_price = data['Y house price of unit area'].mean()
data['affordability'] = (data['Y house price of unit area'] < mean_price).astype(int)
print(data)
# Data nomralization
sc = StandardScaler()
X, y = data.drop(['Y house price of unit area', 'No', 'X5 latitude','X6 longitude','affordability'], axis=1), data['affordability']
X = sc.fit_transform(X)
#Data splittig : 80% train, 20% test
training_amount = int(0.8 * len(data))
X_train = X[:training_amount]
X_test = X[(len(X) - training_amount):]
y_train = y[:training_amount]
y_test = y[(len(X) - training_amount):]

# Initialize parameters
theta_0 = 1
theta = np.zeros(X_train.shape[1])
lr = 0.01
iterations = 1500

def sigmoid(h):
    return 1 / (1 + np.exp(-h))

def compute_loss(y, h):
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / len(y)

def calculate_accuracy(X, y, theta):
    predictions = sigmoid(np.dot(X, theta)) >= 0.5
    return (predictions == y).mean()
def gradient_descent(X_train, y_train, X_test, y_test, theta_0, theta, learning_rate, iterations):
    loss_history_train = []
    loss_history_test = []
    theta_0_history = []
    theta_history = []

    for i in range(iterations):
        l_train = theta_0 + np.dot(X_train, theta)
        h_train = sigmoid(l_train)
        gradient_theta_0 = np.mean(h_train - y_train)
        gradient_theta = np.dot(X_train.T, (h_train - y_train)) / len(y_train)

        theta_0 -= learning_rate * gradient_theta_0
        theta -= learning_rate * gradient_theta

        theta_0_history.append(theta_0)
        theta_history.append(theta.copy())

        loss_train = compute_loss(y_train, h_train)
        loss_history_train.append(loss_train)

        z_test = theta_0 + np.dot(X_test, theta)
        h_test = sigmoid(z_test)
        loss_test = compute_loss(y_test, h_test)
        loss_history_test.append(loss_test)

    return theta_0, theta, theta_0_history, theta_history, loss_history_train, loss_history_test

theta_0, theta, theta_0_history, theta_history, loss_history_train, loss_history_test = gradient_descent(X_train,y_train,X_test, y_test,theta_0, theta, lr, iterations)

training_accuracy = calculate_accuracy(X_train, y_train, theta)
test_accuracy = calculate_accuracy(X_test, y_test, theta)
print(f"Training Accuracy: {training_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# Check for overfitting: training_accuracy
if test_accuracy > training_accuracy:
    print("The model is overfitting.")
else:
    print("The model is not overfitting.")
print()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history_train, label='Training Loss')
plt.plot(loss_history_test, label='Testing Loss', linestyle='--')
plt.title("Loss over iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(theta_0_history, label='Theta0')
''' Plot the convergence of other Thetas as well
for j in range(X_train.shape[1]):
    plt.plot([row[j] for row in theta_history], label=f'Theta {j}')
'''
plt.title('Convergence of Thetas')
plt.xlabel('Iterations')
plt.ylabel('Theta Values')
plt.legend()
plt.tight_layout()
plt.show()
