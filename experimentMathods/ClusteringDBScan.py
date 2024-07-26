from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

#Minmax робить нульовими мінімальні значаення і 1 максимальні.
# Example data: rows are microservices, columns are characteristics (processor load, channel load, RAM load)
data = np.array([
    [0.8, 0.6, 0.7],  # Microservice 1
    [0.5, 0.4, 0.5],  # Microservice 2
    [0.9, 0.8, 0.9],  # Microservice 3
    # Add more microservices as needed
])
plt.scatter(data[:, 0], data[:, 1])
plt.show()
scaler = MinMaxScaler()
print("scaler = " + str(scaler))
normalized_data = scaler.fit_transform(data)
print("norm data = " + str(normalized_data))
###################

similarity_matrix = cosine_similarity(normalized_data)
print("cos sim = " + str(similarity_matrix))
similarity_matrix_data = cosine_similarity(data)
print("cos sim data= " + str(similarity_matrix_data))
######
# Set the parameters for DBSCAN
eps = 0.5  # Maximum distance between two samples for one to be considered as in the neighborhood of the other
min_samples = 2  # Number of samples in a neighborhood for a point to be considered a core point

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(normalized_data)

# Print cluster labels for each microservice (-1 indicates noise)
print(dbscan.labels_)
