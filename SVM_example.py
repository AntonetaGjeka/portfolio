import numpy as np
import matplotlib.pyplot as plt
import cvxopt

# Sample dataset
X = np.array([[2, 3], [3, 3], [3, 2], [5, 3], [6, 4], [2, 6], [3, 5], [6, 6], [7, 7], [8, 6],
              [4, 4], [8, 10], [9, 8], [2, 2], [8, 9], [3, 9], [2, 1], [5, 7], [1, 1], [4, 8]])
y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1])

def linear_svm(X, y):
    m, n = X.shape # Number of samples and features
    y = y.astype(float) # Ensure y is float type
    # Kernel function for linear SVM
    K = np.dot(X, X.T)

    # Constructing the matrices for the QP problem using cvxopt
    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-np.ones(m))
    A = cvxopt.matrix(y, (1, m), tc='d')
    b = cvxopt.matrix(0.0)
    G = cvxopt.matrix(np.diag(-np.ones(m)))
    h = cvxopt.matrix(np.zeros(m))

    # Solve QP problem
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Extracting Lagrange multipliers
    alphas = np.ravel(solution['x'])

    # Support vectors have non zero lagrange multipliers
    sv = alphas > 1e-5
    ind = np.arange(len(alphas))[sv]
    alphas = alphas[sv]
    sv_X = X[sv]
    sv_y = y[sv]

    #Compute the weight vector w
    w = np.sum(alphas[:, None] * sv_y[:, None] * sv_X, axis=0)

    #Compute the intercept b
    b = np.mean(sv_y - np.dot(sv_X, w))
    return w, b, alphas, sv
# Train the SVM (complete the returned values once you write the code)
w, b, alphas, sv = linear_svm(X, y)

def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)

def calculate_error(X, y, w, b):
    misclassified = 0

    # Predict the classes
    predictions = predict(X, w, b)

    # Check misclassifications
    for i in range(len(y)):
        if predictions[i] != y[i]:
            misclassified += 1

    error = misclassified / len(y)
    return error

# Calculate misclassification error
misclassification_error = calculate_error(X, y, w, b)
print(f"Misclassification Error: {misclassification_error}")

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')

# Plot the decision boundary
x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
x2 = -(w[0] * x1 + b) / w[1]
plt.plot(x1, x2, color='black')

# Plot the margin lines
x2_margin1 = -(w[0] * x1 + b - 1) / w[1]
x2_margin2 = -(w[0] * x1 + b + 1) / w[1]
plt.plot(x1, x2_margin1, 'r--')
plt.plot(x1, x2_margin2, 'b--')

# Highlight the support vectors
plt.scatter(X[sv, 0], X[sv, 1], s=100, facecolors='none', edgecolors='k', linewidth=1.5)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Linear SVM with Hard Margin')
plt.show()


