import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# KMeans - підходить для точок, які близько розташовані на площині. Цей метод можна використовувати тільки за умови
# нормування, бо мене цікавить шаблон а не схожість амплітуд, або для визначення схожості амплітуд
# Example data: rows are data points, columns are features
data = np.array([
    [0.8, 0.6],  # Data point 1
    [0.5, 0.4],  # Data point 2
    [0.9, 0.8],  # Data point 3
    [0.4, 0.3],  # Data point 4
    [0.6, 0.5],  # Data point 5
#    [0.2, 0.9],  # Data point 5
])
# Number of clusters
k = 2

centroids = np.array([
    [0.6, 0.5],  # Центроїд 1 (випадково вибраний)
    [0.4, 0.3],  # Центроїд 2 (випадково вибраний)
])

distances = np.zeros((len(data), k))  # Масив для зберігання відстаней
for i in range(len(data)):
    for j in range(k):
        distances[i, j] = np.sqrt(np.sum((data[i] - centroids[j])**2))
print (distances)
######
clusters = [[] for _ in range(k)]  # Список для зберігання точок в кожному кластері
print(clusters)
for i in range(len(data)):
    cluster_idx = np.argmin(distances[i])  # Індекс найближчого кластеру
    print("Search of claster = " + str(i) + " ___" + str(data[i])
          + "____distances=" + str(distances[i])
          + "____argminDist=" + str(np.argmin(distances[i]))
          + "___clust ind = " + str(cluster_idx)
          + "___clusters" + str(clusters[cluster_idx]))
    clusters[cluster_idx].append(data[i])

# Оновимо центроїди як середнє значення точок у кожному кластері
for j in range(k):
    centroids[j] = np.mean(clusters[j], axis=0)
    print("clusters[j]" + str(j) + " ___ " + str(clusters[j]) + " mean = " + str(np.mean(clusters[j], axis=0)))
#####
distances = np.zeros((len(data), k))  # Масив для зберігання відстаней
for i in range(len(data)):
    for j in range(k):
        distances[i, j] = np.sqrt(np.sum((data[i] - centroids[j])**2))
#################


# KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(data)

# Cluster labels
labels = kmeans.labels_

# Centroids
centroids = kmeans.cluster_centers_

# Plotting the clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')
plt.title('KMeans Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()