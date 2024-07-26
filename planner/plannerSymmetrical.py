import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Helper function to add noise to the data and round to integers
def add_noise_and_round(data, noise_level=0.1):
    noisy_data = data + np.random.normal(0, noise_level, data.shape)
    return np.clip(np.round(noisy_data).astype(int), 0, 100)

# Helper function to create gradual peaks
def gradual_peak(length, peak_height, peak_center, peak_width):
    x = np.arange(length)
    return peak_height * np.exp(-((x - peak_center) ** 2) / (2 * peak_width ** 2))

# Generate 24 hours of data
hours = np.arange(24)

# 1. Stably loaded (2 microservices)
stable_load = np.full((2, 24), 50)
stable_load = add_noise_and_round(stable_load, 5)

# 2. Big peak in working hours (4 microservices)
working_hours_peak = np.zeros((4, 24))
for i in range(4):
    working_hours_peak[i] = gradual_peak(24, 80, 13, 4)
working_hours_peak = add_noise_and_round(working_hours_peak, 10)

# 3. Peaks after 12 am and before 6 am (2 microservices)
night_peak = np.zeros((2, 24))
for i in range(2):
    night_peak[i] = gradual_peak(24, 70, 3, 3) + gradual_peak(24, 70, 23, 3)
night_peak = add_noise_and_round(night_peak, 10)

# 4. Peaks from 5 pm till night (2 microservices)
evening_peak = np.zeros((2, 24))
for i in range(2):
    evening_peak[i] = gradual_peak(24, 75, 20, 4)
evening_peak = add_noise_and_round(evening_peak, 10)

# 5. Random peaks (2 microservices)
random_peaks = np.zeros((2, 24))
for i in range(2):
    num_peaks = np.random.randint(3, 6)
    for _ in range(num_peaks):
        peak_center = np.random.randint(0, 24)
        peak_height = np.random.randint(50, 80)
        random_peaks[i] += gradual_peak(24, peak_height, peak_center, 2)
random_peaks = add_noise_and_round(random_peaks, 10)

# Combine all patterns
all_patterns = np.vstack((stable_load, working_hours_peak, night_peak, evening_peak, random_peaks))

# Generate complements (opposite time load)
complements = 100 - all_patterns

# Combine original patterns and complements
microservices = np.vstack((all_patterns, complements))

# Ensure all values are integers between 0 and 100
microservices = np.clip(microservices, 0, 100).astype(int)

# Plot the data with colors for each type of microservice
plt.figure(figsize=(15, 10))
colors = ['blue', 'green', 'red', 'purple', 'orange']
labels = ['Stable', 'Working Hours Peak', 'Night Peak', 'Evening Peak', 'Random Peaks']
for i, (color, label) in enumerate(zip(colors, labels)):
    if i == 0:
        plt.plot(hours, microservices[i], color=color, label=f'{label} (Original)')
        plt.plot(hours, microservices[i+12], color=color, linestyle='--', label=f'{label} (Complement)')
    elif i == 1:
        for j in range(4):
            plt.plot(hours, microservices[i+j], color=color, label=f'{label} (Original)' if j == 0 else '')
            plt.plot(hours, microservices[i+j+12], color=color, linestyle='--', label=f'{label} (Complement)' if j == 0 else '')
    else:
        for j in range(2):
            plt.plot(hours, microservices[i*2+j+2], color=color, label=f'{label} (Original)' if j == 0 else '')
            plt.plot(hours, microservices[i*2+j+14], color=color, linestyle='--', label=f'{label} (Complement)' if j == 0 else '')

plt.title('Processor Load for 24 Microservices')
plt.xlabel('Hour of Day')
plt.ylabel('Processor Load (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Save the data to a file
np.savetxt('microservices_load.csv', microservices, delimiter=',', fmt='%d')
print("Data saved to 'microservices_load.csv'")

# Print the first few rows of the data
print("\nFirst few rows of the data:")
print(microservices[:24])

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