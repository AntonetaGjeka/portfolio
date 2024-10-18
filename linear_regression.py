import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load Iris dataset from scikit-learn
iris = load_iris()

# Define a threshold for MSE
threshold = 0.01

# Convert the Iris dataset to a Pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Extract petal length and petal width
x = df['petal length (cm)'].values  # Petal length
y = df['petal width (cm)'].values  # Petal width

# Data Splitting 80-training, 20-testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize parameters
theta = [[10],[10]]
learning_rate = 0.1  # Learning rate
num_iterations = 300  # Number of iterations
loss_history = []
# Gradient Descent Algorithm
for i in range(num_iterations):
    m = len(y)
    # Compute predictions
    y_pred_train = theta[0] + theta[1] * x_train

    # Compute gradients
    d_theta_0 = np.mean(y_pred_train - y_train) # Gradient nach theta_0 = h0-y
    d_theta_1 = np.mean((y_pred_train - y_train) * x_train) # Gradient nach theta_1 = (h0-y)*x1

    # Update each theta parameter -> Siehe Slide 11
    theta[0] -= learning_rate * d_theta_0
    theta[1] -= learning_rate * d_theta_1

    #Update values for loss:
    loss_history.append((1 / (2 * m)) * np.sum(np.square(y_pred_train - y_train)))
# Model Evaluation
y_pred_train = theta[0] + theta[1] * x_train
mse_train = mean_squared_error(y_train, y_pred_train) # Is this allowed?
print("Train MSE:", mse_train)

y_pred_test = theta[0] + theta[1] * x_test
mse_test = mean_squared_error(y_test, y_pred_test)
print("Test MSE:", mse_test)

# Additional Output
print("Equation: Y = {:.2f}*X + {:.2f} + noise".format(theta[1][0], theta[0][0]))

# Determine if the predicted hypothesis is any good:
if abs(mse_train-mse_test) < threshold:
    print("The predicted hypothesis is good.")
else:
    print("The predicted hypothesis is not good.")

# Visualization of the data and linear regression
sns.scatterplot(x=x_train, y=y_train, color='blue', label='Training Data', alpha=0.5)
sns.lineplot(x=x_train, y=y_pred_train.flatten(), color='red', label='Regression Line')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Linear Regression Model (Iris Dataset)')
plt.legend()
plt.show()

# Plotting the loss history
plt.plot(range(num_iterations), loss_history, color='blue')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss History')
plt.show()

