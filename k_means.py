import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = data.data

def initialize_centroids(X, K):
    #Randomly initialize centroids
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], K, replace=False)
    return X[indices]

def calculate_distances(X, centroids):
    #Calculate the Euclidean distance from each point to each centroid
    distances = np.zeros((X.shape[0], len(centroids)))
    for i, centroid in enumerate(centroids):
        distances[:, i] = np.linalg.norm(X - centroid, axis=1)
    return distances


def assign_clusters(distances):
    #Assign each data point to the nearest centroid
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, K):
    #Update centroids by calculating the mean of all points in each cluster.
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        centroids[k] = X[labels == k].mean(axis=0)
    return centroids


def kmeans(X, K, max_iters=100, tol=1e-4):
    #K-Means clustering algorithm
    centroids = initialize_centroids(X, K)
    sse = []

    for _ in range(max_iters):
        distances = calculate_distances(X, centroids)
        labels = assign_clusters(distances)
        new_centroids = update_centroids(X, labels, K)

        # Calculate SSE
        current_sse = np.sum((X - centroids[labels]) ** 2)
        sse.append(current_sse)

        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return centroids, labels, sse


def calculate_sse(X, labels, centroids):
    #Calculate the sum of squared errors for the clusters
    sse = 0
    for k in range(len(centroids)):
        sse += np.sum((X[labels == k] - centroids[k]) ** 2)
    return sse


def kmeans_multiple_runs(X, K, n_runs=10, max_iters=100):
    #Run K-Means multiple times and return the best result based on SSE
    best_centroids = None
    best_labels = None
    best_sse = float('inf')

    for _ in range(n_runs):
        centroids, labels, sse = kmeans(X, K, max_iters)
        current_sse = calculate_sse(X, labels, centroids)
        if current_sse < best_sse:
            best_sse = current_sse
            best_centroids = centroids
            best_labels = labels

    return best_centroids, best_labels, best_sse


def plot_sse_convergence(sse):
    #Plot the SSE against the number of iterations
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(sse)), sse, marker='o')
    plt.xlabel('Number of Iterations')
    plt.ylabel('SSE')
    plt.title('SSE Convergence')
    plt.grid(True)
    plt.show()


def plot_clusters(X, labels, centroids):
    """Plot the data points with cluster assignments and centroids."""
    plt.figure(figsize=(8, 6))

    # Define color map for the clusters
    colors = ['r', 'g', 'b']

    # Plot each cluster
    for k in range(len(centroids)):
        cluster_points = X[labels == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, c=colors[k], label=f'Cluster {k + 1}')

    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='yellow', marker='X', edgecolor='black', label='Centroids')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Clustering')
    plt.legend()
    plt.show()

# Parameters
K = 3
n_runs = 10
max_iters = 100

# Run K-Means and get the best result
best_centroids, best_labels, best_sse = kmeans_multiple_runs(X, K, n_runs, max_iters)
centroids, labels, sse = kmeans(X, K, max_iters)

# Plot SSE convergence
plot_sse_convergence(sse)

# Print the best SSE, centroids and labels
print("Best SSE:", best_sse)
print("Best Centroids:\n", best_centroids)
print("Best Labels:\n", best_labels)

# Plot the data with cluster assignments
plot_clusters(X, best_labels, best_centroids)
