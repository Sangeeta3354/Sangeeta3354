import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Generate synthetic data for clustering (you can replace this with your own data)
np.random.seed(0)
n_samples = 20
X = np.random.rand(n_samples, 2)

# Perform hierarchical clustering
linked = linkage(X, method='ward')

# Create a dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()