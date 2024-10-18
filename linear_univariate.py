import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Generating synthetic data
np.random.seed(0)
noise = np.random.randn(100, 1) #create a set of random noises
X = 2 * np.random.rand(100, 1)  # One feature
y = 4 + 3 * X + noise  # Linear relationship with noise

# Normalize input data
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Scale X
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=2) #split train & test data, ratio 80:20

# Initialize the hyper-parameters
theta = np.random.randn(1)
theta_0 = np.random.randn()
alpha = 0.01  # Adjust the learning rate
iteration = 1000
#definition of gradientDescent function
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
# Gradient descent
theta, theta_0, loss = gradientDescent(theta, theta_0, alpha, X_train, Y_train, iteration)

#print equation and loss:
print("Equation: Y = {:.2f}*X + {:.2f} + {:.2f}".format(theta[0][0], theta_0, np.std(noise)))
print("loss :" + str((loss[-1])))

# Predict using trained theta and theta_0
Y_pred = theta_0 + X_test.dot(theta)

plt.figure(figsize=(10, 5))
# Plotting Predicted vs Actual
plt.subplot(1, 2, 1)
plt.scatter(X_test, Y_test, color='blue', label='Actual', alpha=0.5)
plt.plot(X_test, Y_pred, color='red', label='Pred.')
plt.title('Predicted vs Actual')

# Plotting Loss History
plt.subplot(1, 2, 2)
plt.plot(range(iteration), loss, color='blue', label='loss-iteration curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss History')
plt.tight_layout()
plt.show()
