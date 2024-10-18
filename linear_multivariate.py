import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
#Generate random data, 3 independ. values(x1,x2,x3) and additional noise
X, y = datasets.make_regression(n_samples=180, n_features=3, noise=15, random_state=4)

# Normalize input data
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Scale X
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Initialize the hyper-parameters
theta = np.random.randn(X_train.shape[1])
theta_0 = 1
alpha = 0.01  # Adjust the learning rate
iteration = 1000

def gradientDescent(theta, theta_0, alpha, X, y, iteration):
    m = len(y)
    loss_history = []
    for i in range(iteration):
        predict = theta_0 + X.dot(theta)
        loss = (1 / (2 * m)) * np.sum(np.square(predict - y))  # Calculate loss
        loss_history.append(loss)  # Append to loss history
        theta = theta - alpha * (1 / m) * (X.T.dot(predict - y))
        theta_0 = theta_0 - alpha * (1 / m) * (np.sum(predict - y))
    return theta, theta_0, loss_history
#Call the gradientDescent function to minimize loss
theta, theta_0, loss = gradientDescent(theta, theta_0, alpha, X_train, Y_train, iteration)
#print out the parameters of the hypothesis
print("first parameter: " + str(theta))
print("second parameter: " + str(theta_0) + "\n")
print("loss :" + str((loss[-1])))

# Predict using trained theta and theta_0
Y_pred = theta_0 + X_test.dot(theta)

plt.figure(figsize=(10, 5))
# Plotting Predicted vs Actual
plt.subplot(1, 2, 1)
plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Actual')
plt.scatter(range(len(Y_test)), Y_pred, color='red', label='Predicted')
plt.plot(range(len(Y_test)), Y_pred, color='green', label='Line of Best Fit')
plt.title('Predicted vs Actual')
plt.legend()

# Plotting Loss History
plt.subplot(1, 2, 2)
plt.plot(range(iteration),loss, color='blue', label='loss-iteration curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss History')
plt.tight_layout()
plt.show()
