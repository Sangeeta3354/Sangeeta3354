import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generate synthetic data for clustering (you can replace this with your own data)
n_samples = 200
X, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=0)

# Initialize the DBSCAN model
dbscan = DBSCAN(eps=0.3, min_samples=5)

# Fit the model to the data
dbscan.fit(X)

# Get cluster labels (-1 indicates noise)
labels = dbscan.labels_

# Plot the data points and cluster assignments
unique_labels = np.unique(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    if label == -1:
        color = 'gray'  # Noise points are gray
    class_member_mask = (labels == label)
    xy = X[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[color], label=f'Cluster {label}')

plt.legend()
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()