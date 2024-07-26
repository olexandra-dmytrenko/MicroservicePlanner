import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)

# Generate 24 hours of data
hours = np.arange(24)

# Helper function to add noise and round to integers
def add_noise_and_round(data, noise_level=0.1):
    noisy_data = data + np.random.normal(0, noise_level * np.max(data), data.shape)
    return np.clip(np.round(noisy_data).astype(int), 0, 100)

# Helper function to create gradual peaks
def gradual_peak(length, peak_height, peak_center, peak_width):
    x = np.arange(length)
    return peak_height * np.exp(-((x - peak_center) ** 2) / (2 * peak_width ** 2))

# Function to generate a random microservice load pattern
def generate_pattern(pattern_type):
    base_load = np.random.randint(20, 40)
    pattern = np.full(24, base_load, dtype=float)

    if pattern_type == 'stable':
        pattern += np.random.randint(5, 15)
    elif pattern_type == 'work_hours':
        pattern += gradual_peak(24, 50, 13, 4)
    elif pattern_type == 'night':
        pattern += gradual_peak(24, 40, 3, 3) + gradual_peak(24, 40, 23, 3)
    elif pattern_type == 'evening':
        pattern += gradual_peak(24, 45, 20, 4)
    elif pattern_type == 'random':
        num_peaks = np.random.randint(2, 4)
        for _ in range(num_peaks):
            peak_center = np.random.randint(0, 24)
            peak_height = np.random.randint(30, 50)
            peak_width = np.random.uniform(2, 4)
            pattern += gradual_peak(24, peak_height, peak_center, peak_width)

    return add_noise_and_round(pattern, noise_level=0.1)

# Generate microservices data
patterns = ['stable'] * 2 + ['work_hours'] * 4 + ['night'] * 2 + ['evening'] * 2 + ['random'] * 2
microservices = np.array([generate_pattern(p) for p in patterns])

# Generate complementary patterns with noise
complementary = add_noise_and_round(100 - microservices, noise_level=0.15)

# Combine original patterns and complements
all_microservices = np.vstack((microservices, complementary))

# Ensure all values are integers between 0 and 100
all_microservices = np.clip(all_microservices, 0, 100).astype(int)

# Reshape the data to be 2D: (n_samples, n_features)
microservices_2d = all_microservices.reshape(all_microservices.shape[0], -1)

# Apply Z-normalization
scaler = StandardScaler()
microservices_normalized = scaler.fit_transform(microservices_2d)

# Perform K-means clustering
n_clusters = 7  # You can adjust this number
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(microservices_normalized)

# Apply PCA for visualization
pca = PCA(n_components=2)
microservices_pca = pca.fit_transform(microservices_normalized)
centroids_pca = pca.transform(kmeans.cluster_centers_)

# Plot the clustered data points and centroids
plt.figure(figsize=(12, 8))
colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

for i, color in enumerate(colors):
    cluster_points = microservices_pca[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f'Cluster {i}', alpha=0.7)
    plt.scatter(centroids_pca[i, 0], centroids_pca[i, 1], color=color, marker='x', s=200, linewidths=3)

plt.title('Microservices Clustered (PCA Visualization)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot original time series data with cluster colors
plt.figure(figsize=(15, 10))

for i in range(all_microservices.shape[0]):
    plt.plot(hours, all_microservices[i], color=colors[cluster_labels[i]],
             label=f'Cluster {cluster_labels[i]}' if cluster_labels[i] not in cluster_labels[:i] else "")

# Plot cluster centroids
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
for i, centroid in enumerate(centroids_original):
    plt.plot(hours, centroid, color=colors[i], linewidth=3, linestyle='--',
             label=f'Centroid {i}')

plt.title('Processor Load for 24 Microservices (K-means Clustered)')
plt.xlabel('Hour of Day')
plt.ylabel('Processor Load (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Print cluster information
for i in range(n_clusters):
    cluster_members = np.where(cluster_labels == i)[0]
    print(f"\nCluster {i}:")
    print(f"  Members: {cluster_members}")
    print(f"  Centroid: {centroids_original[i]}")

# Save the normalized data and cluster labels to files
np.savetxt('microservices_load_normalized.csv', microservices_normalized, delimiter=',')
np.savetxt('cluster_labels.csv', cluster_labels, delimiter=',', fmt='%d')
print("\nNormalized data saved to 'microservices_load_normalized.csv'")
print("Cluster labels saved to 'cluster_labels.csv'")






#-------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# [Previous code for generating microservices data remains the same]
# ... [Include all the code up to the generation of 'microservices' array]

# Reshape the data to be 2D: (n_samples, n_features)
microservices_2d = microservices.reshape(microservices.shape[0], -1)

# Apply Z-normalization
scaler = StandardScaler()
microservices_normalized = scaler.fit_transform(microservices_2d)

# Perform K-means clustering
n_clusters = 7  # You can adjust this number
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(microservices_normalized)

# Apply PCA for visualization
pca = PCA(n_components=2)
microservices_pca = pca.fit_transform(microservices_normalized)
centroids_pca = pca.transform(kmeans.cluster_centers_)

# Plot the clustered data points and centroids
plt.figure(figsize=(12, 8))
colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

for i, color in enumerate(colors):
    cluster_points = microservices_pca[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f'Cluster {i}', alpha=0.7)
    plt.scatter(centroids_pca[i, 0], centroids_pca[i, 1], color=color, marker='x', s=200, linewidths=3)

plt.title('Microservices Clustered (PCA Visualization)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot original time series data with cluster colors
plt.figure(figsize=(15, 10))

for i in range(microservices.shape[0]):
    plt.plot(hours, microservices[i], color=colors[cluster_labels[i]],
             label=f'Cluster {cluster_labels[i]}' if cluster_labels[i] not in cluster_labels[:i] else "")

# Plot cluster centroids
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
for i, centroid in enumerate(centroids_original):
    plt.plot(hours, centroid, color=colors[i], linewidth=3, linestyle='--',
             label=f'Centroid {i}')

plt.title('Original Processor Load for 24 Microservices (K-means Clustered)')
plt.xlabel('Hour of Day')
plt.ylabel('Processor Load (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Print cluster information
for i in range(n_clusters):
    cluster_members = np.where(cluster_labels == i)[0]
    print(f"\nCluster {i}:")
    print(f"  Members: {cluster_members}")
    print(f"  Centroid: {centroids_original[i]}")

# Save the normalized data and cluster labels to files
np.savetxt('microservices_load_normalized.csv', microservices_normalized, delimiter=',')
np.savetxt('cluster_labels.csv', cluster_labels, delimiter=',', fmt='%d')
print("\nNormalized data saved to 'microservices_load_normalized.csv'")
print("Cluster labels saved to 'cluster_labels.csv'")