import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple
import matplotlib.pyplot as plt

class TimeSeriesBalancer:
    def __init__(self, time_series: List[List[float]], num_slots: int):
        self.time_series = np.array(time_series)
        self.num_series = len(time_series)
        self.num_slots = num_slots

    def kmeans_clustering(self, n_clusters: int) -> Tuple[List[List[int]], float]:
        """Кластеризація часових рядів за допомогою KMeans"""
        # Застосовуємо KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.time_series)

        # Формуємо групи
        groups = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(cluster_labels):
            groups[label].append(idx)

        # Обчислюємо якість розподілу
        variance = self._calculate_total_variance(groups)

        return groups, variance

    def greedy_balancing(self) -> Tuple[List[List[int]], float]:
        """Жадібний алгоритм для балансування навантаження"""
        # Обчислюємо "вагу" кожного часового ряду (сума всіх значень)
        weights = np.sum(self.time_series, axis=1)

        # Сортуємо індекси за спаданням ваги
        sorted_indices = np.argsort(-weights)

        # Ініціалізуємо групи
        groups = []
        group_loads = []  # Зберігаємо сумарне навантаження по часових слотах для кожної групи

        # Розподіляємо часові ряди по групах
        for idx in sorted_indices:
            series = self.time_series[idx]

            # Якщо груп ще немає, створюємо першу
            if not groups:
                groups.append([idx])
                group_loads.append(series)
                continue

            # Знаходимо найкращу групу для поточного часового ряду
            best_variance = float('inf')
            best_group_idx = 0

            # Пробуємо додати до існуючих груп
            for i, group_load in enumerate(group_loads):
                # Обчислюємо нове навантаження групи
                new_load = group_load + series
                variance = np.var(new_load)

                if variance < best_variance:
                    best_variance = variance
                    best_group_idx = i

            # Перевіряємо, чи краще створити нову групу
            new_group_variance = np.var(series)
            if new_group_variance < best_variance:
                groups.append([idx])
                group_loads.append(series)
            else:
                groups[best_group_idx].append(idx)
                group_loads[best_group_idx] += series

        variance = sum(np.var(load) for load in group_loads)
        return groups, variance

    def knapsack_balancing(self, target_load: float) -> Tuple[List[List[int]], float]:
        """Балансування на основі задачі пакування рюкзака"""
        groups = []
        remaining_indices = set(range(self.num_series))

        while remaining_indices:
            current_group = []
            current_load = np.zeros(self.num_slots)

            # Знаходимо часові ряди, які найкраще підходять до цільового навантаження
            while remaining_indices:
                best_idx = None
                best_diff = float('inf')

                for idx in remaining_indices:
                    new_load = current_load + self.time_series[idx]
                    diff = np.mean(np.abs(new_load - target_load))

                    if diff < best_diff:
                        best_diff = diff
                        best_idx = idx

                # Якщо додавання погіршує ситуацію, починаємо нову групу
                new_load = current_load + self.time_series[best_idx]
                if np.mean(new_load) > target_load * 1.2:  # 20% допуск
                    break

                current_group.append(best_idx)
                current_load = new_load
                remaining_indices.remove(best_idx)

            if current_group:
                groups.append(current_group)

        variance = self._calculate_total_variance(groups)
        return groups, variance

    def _calculate_total_variance(self, groups: List[List[int]]) -> float:
        """Обчислює загальну дисперсію для всіх груп"""
        return sum(np.var(np.sum(self.time_series[group], axis=0)) for group in groups)

    def visualize_groups(self, groups: List[List[int]], title: str = ""):
        """Візуалізує розподіл навантаження по групах"""
        num_groups = len(groups)
        fig, axes = plt.subplots(num_groups, 1, figsize=(12, 4*num_groups))
        if num_groups == 1:
            axes = [axes]

        for i, group in enumerate(groups):
            group_sum = np.sum(self.time_series[group], axis=0)
            axes[i].plot(range(self.num_slots), group_sum, 'b-', label=f'Group {i+1}')
            axes[i].set_title(f'Group {i+1} (Series indices: {group})')
            axes[i].set_xlabel('Time slot')
            axes[i].set_ylabel('Load')
            axes[i].grid(True)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

# Приклад використання
if __name__ == "__main__":
    # Тестові дані (6 часових слотів)
    time_series = [
        [0, 0, 3, 8, 5, 0],
        [0, 4, 0, 0, 0, 0],
        [2, 3, 4, 0, 0, 1],
        [0, 0, 0, 5, 4, 3],
        [1, 1, 1, 1, 1, 1],
        [0, 2, 4, 6, 4, 2],
    ]

    balancer = TimeSeriesBalancer(time_series, num_slots=6)

    # KMeans підхід
#    kmeans_groups, kmeans_variance = balancer.kmeans_clustering(n_clusters=2)
 #   print("\nKMeans clustering:")
  #  print("Groups:", kmeans_groups)
   # print("Variance:", kmeans_variance)
    #balancer.visualize_groups(kmeans_groups, "KMeans Clustering")

    # Жадібний алгоритм
    # greedy_groups, greedy_variance = balancer.greedy_balancing()
    # print("\nGreedy balancing:")
    # print("Groups:", greedy_groups)
    # print("Variance:", greedy_variance)
    # balancer.visualize_groups(greedy_groups, "Greedy Algorithm")

    # Підхід на основі пакування рюкзака
    target_load = 11
    #target_load = np.mean([np.sum(series) for series in time_series])
    knapsack_groups, knapsack_variance = balancer.knapsack_balancing(target_load)
    print("\nKnapsack balancing:")
    print("Groups:", knapsack_groups)
    print("Variance:", knapsack_variance)
    print("Target Load:", target_load)
    balancer.visualize_groups(knapsack_groups, "Knapsack Approach")