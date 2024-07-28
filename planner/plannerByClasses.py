import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class MicroserviceDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.hours = np.arange(24)
        self.microservices = None

    def add_noise_and_round(self, data, noise_level=0.1):
        noisy_data = data + np.random.normal(0, noise_level, data.shape)
        return np.clip(np.round(noisy_data).astype(int), 0, 100)

    def gradual_peak(self, length, peak_height, peak_center, peak_width):
        x = np.arange(length)
        return peak_height * np.exp(-((x - peak_center) ** 2) / (2 * peak_width ** 2))

    def generate_data(self):
        stable_load = np.full((2, 24), 50)
        stable_load = self.add_noise_and_round(stable_load, 5)

        working_hours_peak = np.zeros((4, 24))
        for i in range(4):
            working_hours_peak[i] = self.gradual_peak(24, 80, 13, 4)
        working_hours_peak = self.add_noise_and_round(working_hours_peak, 10)

        night_peak = np.zeros((2, 24))
        for i in range(2):
            night_peak[i] = self.gradual_peak(24, 70, 3, 3) + self.gradual_peak(24, 70, 23, 3)
        night_peak = self.add_noise_and_round(night_peak, 10)

        evening_peak = np.zeros((2, 24))
        for i in range(2):
            evening_peak[i] = self.gradual_peak(24, 75, 20, 4)
        evening_peak = self.add_noise_and_round(evening_peak, 10)

        random_peaks = np.zeros((2, 24))
        for i in range(2):
            num_peaks = np.random.randint(3, 6)
            for _ in range(num_peaks):
                peak_center = np.random.randint(0, 24)
                peak_height = np.random.randint(50, 80)
                random_peaks[i] += self.gradual_peak(24, peak_height, peak_center, 2)
        random_peaks = self.add_noise_and_round(random_peaks, 10)

        all_patterns = np.vstack((stable_load, working_hours_peak, night_peak, evening_peak, random_peaks))
        complements = 100 - all_patterns
        self.microservices = np.vstack((all_patterns, complements))
        self.microservices = np.clip(self.microservices, 0, 100).astype(int)

        return self.microservices

    def visualize_data(self):
        if self.microservices is None:
            raise ValueError("Data not generated yet. Call generate_data() first.")

        plt.figure(figsize=(15, 10))
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        labels = ['Stable', 'Working Hours Peak', 'Night Peak', 'Evening Peak', 'Random Peaks']

        for i, (color, label) in enumerate(zip(colors, labels)):
            if i == 0:
                plt.plot(self.hours, self.microservices[i], color=color, label=f'{label} (Original)')
                plt.plot(self.hours, self.microservices[i + 12], color=color, linestyle='--',
                         label=f'{label} (Complement)')
            elif i == 1:
                for j in range(4):
                    plt.plot(self.hours, self.microservices[i + j], color=color,
                             label=f'{label} (Original)' if j == 0 else '')
                    plt.plot(self.hours, self.microservices[i + j + 12], color=color, linestyle='--',
                             label=f'{label} (Complement)' if j == 0 else '')
            else:
                for j in range(2):
                    plt.plot(self.hours, self.microservices[i * 2 + j + 2], color=color,
                             label=f'{label} (Original)' if j == 0 else '')
                    plt.plot(self.hours, self.microservices[i * 2 + j + 14], color=color, linestyle='--',
                             label=f'{label} (Complement)' if j == 0 else '')

        plt.title('Processor Load for 24 Microservices')
        plt.xlabel('Hour of Day')
        plt.ylabel('Processor Load (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class MicroserviceData:
    def __init__(self, timeline):
        self.timeline = timeline
        self.max_value = np.max(timeline)
        self.min_value = np.min(timeline)
        self.range = self.max_value - self.min_value


class Cluster:
    def __init__(self, centroid, centroid_normalized, normalized_microservices, microservice_data):
        self.centroid = centroid
        self.centroid_normalized = centroid_normalized
        self.complementary_centroid = self.find_symmetrical_vector(centroid_normalized)
        self.normalized_microservices = normalized_microservices
        self.microservice_data = microservice_data

    def find_symmetrical_vector(self, vector):
        def symmetrical_value(x):
            if x < 0:
                return x + 2
            elif x > 0:
                return x - 2
            else:
                return 0

        # numpy's vectorize to apply the function to each element
        symmetrical_func = np.vectorize(symmetrical_value)
        return symmetrical_func(vector)


class MicroserviceSimilarityFinder:
    def __init__(self, microservices_data):
        self.microservices_data = [MicroserviceData(timeline) for timeline in microservices_data]
        self.microservices_2d = microservices_data.reshape(microservices_data.shape[0], -1)
        self.scaler = StandardScaler()
        self.kmeans = None
        self.normalized_data = None
        self.cluster_labels = None
        self.hours = np.arange(24)

    def normalize_data(self):
        self.normalized_data = self.scaler.fit_transform(self.microservices_2d)
        return self.normalized_data

    def cluster_data(self, n_clusters=7):
        if self.normalized_data is None:
            self.normalize_data()

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_labels = self.kmeans.fit_predict(self.normalized_data)
        return self.cluster_labels

    def get_clusters(self):
        if self.cluster_labels is None:
            self.cluster_data()

        clusters = []
        centroids_original = self.scaler.inverse_transform(self.kmeans.cluster_centers_)

        for i in range(len(self.kmeans.cluster_centers_)):
            cluster_indices = np.where(self.cluster_labels == i)[0]
            normalized_microservices = self.normalized_data[cluster_indices]
            microservice_data = [self.microservices_data[j] for j in cluster_indices]
            centroid = centroids_original[i]
            centroid_normalized = self.kmeans.cluster_centers_[i]
            clusters.append(Cluster(centroid, centroid_normalized, normalized_microservices, microservice_data))

        return clusters

    def visualize_clusters_pca(self):
        if self.normalized_data is None or self.cluster_labels is None:
            self.cluster_data()

        pca = PCA(n_components=2)
        microservices_pca = pca.fit_transform(self.normalized_data)
        centroids_pca = pca.transform(self.kmeans.cluster_centers_)

        plt.figure(figsize=(12, 8))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.kmeans.cluster_centers_)))

        for i, color in enumerate(colors):
            cluster_points = microservices_pca[self.cluster_labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f'Cluster {i}', alpha=0.7)
            plt.scatter(centroids_pca[i, 0], centroids_pca[i, 1], color=color, marker='x', s=200, linewidths=3)

        plt.title('Microservices Clustered (PCA Visualization)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def visualize_clusters_original(self):
        if self.cluster_labels is None:
            self.cluster_data()

        plt.figure(figsize=(15, 10))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.kmeans.cluster_centers_)))

        for i, microservice_data in enumerate(self.microservices_data):
            plt.plot(self.hours, microservice_data.timeline, color=colors[self.cluster_labels[i]],
                     label=f'Cluster {self.cluster_labels[i]}' if self.cluster_labels[i] not in self.cluster_labels[
                                                                                                :i] else "")

        centroids_original = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        for i, centroid in enumerate(centroids_original):
            plt.plot(self.hours, centroid, color=colors[i], linewidth=3, linestyle='--',
                     label=f'Centroid {i}')

        plt.title('Original Processor Load for Microservices (K-means Clustered)')
        plt.xlabel('Hour of Day')
        plt.ylabel('Processor Load (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


################

import numpy as np
from scipy.spatial.distance import euclidean

import numpy as np
from sklearn.cluster import KMeans


class ComplementaryClusterFinder:
    def __init__(self, clusters, centroid_discrepancy_threshold=30, amplitude_discrepancy_threshold=30):
        self.clusters = clusters
        self.centroid_discrepancy_threshold = centroid_discrepancy_threshold
        self.amplitude_discrepancy_threshold = amplitude_discrepancy_threshold
        self.complementation_stats = []
        self.complementary_microservices = []

    def calculate_discrepancy(self, vector1, vector2):
        sum_max = np.max(vector1 + vector2)
        discrepancy = np.sum(sum_max - (vector1 + vector2)) / len(vector1)
        return discrepancy / sum_max * 100

    def is_complementary_centroid_match(self, cluster1, cluster2):
        kmeans = KMeans(n_clusters=1, random_state=42)
        combined_data = np.vstack((cluster1.normalized_microservices, cluster2.complementary_centroid))
        labels = kmeans.fit_predict(combined_data)
        return 0 if labels[-1] == labels[0] else 1

    def find_complementary_pairs(self):
        n_clusters = len(self.clusters)
        for i in range(n_clusters):
            for j in range(i, n_clusters):
                discrepancy = self.calculate_discrepancy(self.clusters[i].centroid, self.clusters[j].centroid)
                if discrepancy <= self.centroid_discrepancy_threshold:
                    centroid_match = self.is_complementary_centroid_match(self.clusters[i], self.clusters[j])
                    self.complementation_stats.append({
                        'cluster1': i,
                        'cluster2': j,
                        'discrepancy': discrepancy,
                        'centroid_match': centroid_match,
                        'active': True
                    })

        self.complementation_stats.sort(key=lambda x: x['discrepancy'])

    def find_microservice_matches(self):
        for pair in self.complementation_stats:
            if not pair['active']:
                continue

            cluster1 = self.clusters[pair['cluster1']]
            cluster2 = self.clusters[pair['cluster2']]

            i = 0
            while i < len(cluster1.microservice_data):
                ms1 = cluster1.microservice_data[i]
                j = 0
                while j < len(cluster2.microservice_data):
                    ms2 = cluster2.microservice_data[j]

                    range_discrepancy = abs(ms1.range - ms2.range) / max(ms1.range, ms2.range) * 100
                    if range_discrepancy <= self.amplitude_discrepancy_threshold:
                        discrepancy = self.calculate_discrepancy(ms1.timeline, ms2.timeline)
                        if discrepancy <= self.centroid_discrepancy_threshold:
                            self.complementary_microservices.append((ms1, ms2))
                            del cluster1.microservice_data[i]
                            del cluster2.microservice_data[j]
                            i -= 1
                            break
                    j += 1
                i += 1

            if not cluster1.microservice_data or not cluster2.microservice_data:
                pair['active'] = False

    def find_complementary_microservices(self):
        self.find_complementary_pairs()
        self.find_microservice_matches()

    def print_results(self):
        print("Complementation Stats:")
        for stat in self.complementation_stats:
            print(f"Clusters {stat['cluster1']} and {stat['cluster2']}:")
            print(f"  Discrepancy: {stat['discrepancy']:.2f}%")
            print(f"  Centroid Match: {'Yes' if stat['centroid_match'] == 0 else 'No'}")
            print(f"  Active: {'Yes' if stat['active'] else 'No'}")

        print("\nComplementary Microservices:")
        for i, (ms1, ms2) in enumerate(self.complementary_microservices):
            print(f"Pair {i + 1}:")
            print(f"  Microservice 1: Max={ms1.max_value}, Min={ms1.min_value}, Range={ms1.range}")
            print(f"  Microservice 2: Max={ms2.max_value}, Min={ms2.min_value}, Range={ms2.range}")


################
# Generate data
generator = MicroserviceDataGenerator()
microservices_data = generator.generate_data()
# generator.visualize_data()

# Find similarities
similarity_finder = MicroserviceSimilarityFinder(microservices_data)
normalized_data = similarity_finder.normalize_data()
cluster_labels = similarity_finder.cluster_data(n_clusters=7)
clusters = similarity_finder.get_clusters()

# Visualize clusters using PCA
# similarity_finder.visualize_clusters_pca()

# Visualize clusters on original data
# similarity_finder.visualize_clusters_original()

# Access cluster information
for i, cluster in enumerate(clusters):
    print(f"Cluster {i}:")
    print(f"  Centroid: {cluster.centroid}")
    print(f"  Complementary centroid: {cluster.complementary_centroid}")
    print(f"  Number of microservices: {len(cluster.microservice_data)}")
    print(f"  Average range of microservices: {np.mean([ms.range for ms in cluster.microservice_data])}")

# Assuming you have already created clusters using MicroserviceSimilarityFinder
finder = ComplementaryClusterFinder(clusters)
finder.find_complementary_microservices()
finder.print_results()
