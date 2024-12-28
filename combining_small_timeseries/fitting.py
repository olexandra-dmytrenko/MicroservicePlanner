import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

class TimeSeriesBalancer:
    def __init__(self, time_series: List[List[float]], num_slots: int):
        self.time_series = np.array(time_series)
        self.num_series = len(time_series)
        self.num_slots = num_slots

    def calculate_group_variance(self, group_indices: List[int]) -> float:
        """Обчислює дисперсію сумарного навантаження групи по часових слотах"""
        if not group_indices:
            return 0
        group_sum = np.sum(self.time_series[group_indices], axis=0)
        return np.var(group_sum)

    def find_optimal_groups(self, max_groups: int) -> Tuple[List[List[int]], float]:
        """Знаходить оптимальні групи методом локального пошуку"""
        best_groups = []
        best_variance = float('inf')

        # Починаємо з випадкового розподілу
        current_groups = self._initialize_random_groups(max_groups)

        for iteration in range(1000):  # Можна налаштувати кількість ітерацій
            improved = False

            # Спробуємо перемістити кожен часовий ряд в іншу групу
            for series_idx in range(self.num_series):
                current_group_idx = None
                # Знаходимо поточну групу для series_idx
                for i, group in enumerate(current_groups):
                    if series_idx in group:
                        current_group_idx = i
                        break

                original_variance = self._calculate_total_variance(current_groups)
                best_move_variance = original_variance
                best_move = None

                # Пробуємо перемістити в кожну іншу групу
                for new_group_idx in range(len(current_groups)):
                    if new_group_idx != current_group_idx:
                        # Створюємо нову конфігурацію груп
                        new_groups = self._create_new_configuration(
                            current_groups, series_idx,
                            current_group_idx, new_group_idx
                        )

                        new_variance = self._calculate_total_variance(new_groups)

                        if new_variance < best_move_variance:
                            best_move_variance = new_variance
                            best_move = (new_group_idx, new_groups)

                # Якщо знайшли краще рішення, застосовуємо його
                if best_move and best_move_variance < original_variance:
                    current_groups = best_move[1]
                    improved = True

            # Якщо не вдалося покращити рішення, завершуємо
            if not improved:
                break

            current_variance = self._calculate_total_variance(current_groups)
            if current_variance < best_variance:
                best_variance = current_variance
                best_groups = current_groups.copy()

        return best_groups, best_variance

    def _initialize_random_groups(self, max_groups: int) -> List[List[int]]:
        """Ініціалізує випадковий розподіл по групах"""
        groups = [[] for _ in range(max_groups)]
        series_indices = list(range(self.num_series))
        np.random.shuffle(series_indices)

        # Розподіляємо часові ряди по групах рівномірно
        for i, idx in enumerate(series_indices):
            groups[i % max_groups].append(idx)

        return [group for group in groups if group]  # Видаляємо порожні групи

    def _calculate_total_variance(self, groups: List[List[int]]) -> float:
        """Обчислює загальну дисперсію для всіх груп"""
        return sum(self.calculate_group_variance(group) for group in groups)

    def _create_new_configuration(self, groups: List[List[int]],
                                  series_idx: int, from_group_idx: int,
                                  to_group_idx: int) -> List[List[int]]:
        """Створює нову конфігурацію груп, переміщуючи один часовий ряд"""
        new_groups = [group.copy() for group in groups]
        new_groups[from_group_idx].remove(series_idx)
        new_groups[to_group_idx].append(series_idx)
        return new_groups

    def visualize_groups(self, groups: List[List[int]]):
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
        [1, 0, 0, 0, 3, 1],
        [0, 2, 0, 6, 0, 0],
    ]

    balancer = TimeSeriesBalancer(time_series, num_slots=6)
    optimal_groups, variance = balancer.find_optimal_groups(max_groups=3)

    print("Optimal groups:", optimal_groups)
    print("Total variance:", variance)

    balancer.visualize_groups(optimal_groups)