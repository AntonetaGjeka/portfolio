import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data = pd.read_csv("Real_estate.csv")

# Step 2: Dataset splitting & normalizing
sc = StandardScaler()
X, y = data.drop(['X1 transaction date','Y house price of unit area', 'No', 'X5 latitude','X6 longitude'], axis=1), data['Y house price of unit area']
X = sc.fit_transform(X)

training_amount = int(0.8 * len(data))
X_train = X[:training_amount]
X_test = X[(len(X) - training_amount):]
y_train = y[:training_amount]
y_test = y[(len(X) - training_amount):]

# Initialize parameters
lr = 0.01
iteration = 1000
features = X_train.shape[1]
regularization_param = [20, 10, 5, 0.5, 0]

#Gradient descent function definition:
def gradientDescent(features, theta_0, theta, learning_rate, regularization_param, X_train, y_train, X_test, y_test, iteration):
    #Define Variables where the history (loss & convergence will be stored)
    loss_history_train = []
    loss_history_test = []
    theta_0_history = []
    theta_history = []
    for i in range(iteration):
        #For train data:
        predict_train = theta_0 + np.dot(X_train, theta) #Calculate predicted value
        loss_train = (1 / (2 * len(y_train))) * np.sum(np.square(predict_train - y_train)) #Ccalculate loss according to formula
        loss_history_train.append(loss_train) # Save the value for plotting purposes

        #For test data:
        predict_test = theta_0 + np.dot(X_test, theta) #Calculate predicted value for the test data
        loss_test = (1 / (2 * len(y_test))) * np.sum(np.square(predict_test - y_test)) #How much is the loss?
        loss_history_test.append(loss_test)

        #Update the theta parameters - with regularization
        theta_0 = theta_0 - learning_rate * (1 / len(y_train)) * np.sum(predict_train - y_train)
        for j in range(features):
            gradient = np.dot((predict_train - y_train), X_train[:, j])
            theta[j] = theta[j] * (1 - learning_rate * (regularization_param / len(y_train))) - learning_rate * (gradient/len(y_train))

        #Save the theta values for plotting purposes:
        theta_history.append(theta.copy())
        theta_0_history.append(theta_0.copy())

    return theta_0, theta, loss_history_train, loss_history_test, theta_0_history, theta_history

for reg in regularization_param:
    theta_0 = 1
    theta = np.ones(X_train.shape[1])
    theta_0, theta, loss_history_train, loss_history_test, theta_0_history, theta_history = (
        gradientDescent(features, theta_0, theta, lr, reg, X_train, y_train, X_test, y_test, iteration))

    plt.figure(figsize=(10, 5))
    # Plotting train vs test loss over iterations
    plt.subplot(1, 2, 1)
    plt.plot(range(0, iteration), loss_history_train, label="train loss")
    plt.plot(range(0, iteration), loss_history_test, label="test loss")
    plt.title("Loss over time")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()

    print(f"Regularization: {reg}")
    print('MSE_train = ', loss_history_train[len(loss_history_train) - 1])
    print('MSE_test = ', loss_history_test[len(loss_history_test) - 1])

    #plotting the theta-parameters convergence:
    plt.subplot(1, 2, 2)
    for j in range(features):
        plt.plot([row[j] for row in theta_history], label=f'Theta {j}')
    plt.title(f"Parameter Convergence with Regularization: {reg}")
    plt.xlabel("Iteration")
    plt.ylabel("Parameter Value")
    plt.legend()
    plt.tight_layout()
    plt.show()