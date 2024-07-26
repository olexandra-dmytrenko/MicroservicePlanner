from sklearn.cluster import KMeans
import numpy as np

# Sample data
data = np.array([
    [1, 2], [1, 4], [1, 0],
    [4, 2], [4, 4], [4, 0]
])

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
print(kmeans.labels_)
print(kmeans.cluster_centers_)

