import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

def group_similar_microservices(resource_usage_matrix):
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(resource_usage_matrix)
    print(similarity_matrix)
    # Perform clustering (example with KMeans)
    kmeans = KMeans(n_clusters=3).fit(resource_usage_matrix)
    clusters = kmeans.labels_

    return clusters

# Example usage with dummy data
cpu_usage = np.array([[10, 20, 30], [12, 22, 32], [50, 60, 70]])
clusters = group_similar_microservices(cpu_usage)
print(clusters)
