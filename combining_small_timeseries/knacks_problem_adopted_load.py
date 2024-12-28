import numpy as np
import matplotlib.pyplot as plt

class AdaptiveLoadBalancer:
    def __init__(self, time_series: list, max_deviation: float = 0.3,
                 growth_factor: float = 1.5):
        self.time_series = np.array(time_series)
        self.num_series = len(time_series)
        self.num_slots = len(time_series[0])
        self.max_deviation = max_deviation
        self.growth_factor = growth_factor

        # Обчислюємо базове цільове навантаження
        all_loads = np.sum(self.time_series, axis=1)
        self.base_target = np.mean(all_loads)
        print(f"Базове цільове навантаження: {self.base_target:.2f}")

    def check_load_balance(self, loads: np.ndarray, iteration: int) -> bool:
        """
        Перевіряє чи відхилення між годинами в межах допустимого
        з урахуванням поточної ітерації
        """
        if np.sum(loads) == 0:
            return True

        non_zero_loads = loads[loads > 0]
        if len(non_zero_loads) == 0:
            return True

        # Розраховуємо допустиме відхилення з урахуванням ітерації
        current_max_deviation = self.max_deviation * (1 + iteration * 0.1)

        max_load = np.max(non_zero_loads)
        min_load = np.min(non_zero_loads)

        relative_diff = (max_load - min_load) / max_load
        return relative_diff <= current_max_deviation

    def get_target_load(self, iteration: int) -> float:
        """
        Розраховує цільове навантаження з урахуванням ітерації
        """
        return self.base_target * (self.growth_factor ** iteration)

    def find_best_series(self, current_load: np.ndarray,
                         remaining_indices: set,
                         iteration: int) -> tuple:
        """Знаходить найкращий часовий ряд для додавання в групу"""
        best_idx = None
        best_score = float('inf')
        best_load = None

        target_load = self.get_target_load(iteration)
        current_sum = np.sum(current_load)

        # Якщо група порожня, вибираємо ряд з навантаженням близьким до цільового
        if current_sum == 0:
            best_diff = float('inf')
            for idx in remaining_indices:
                series_sum = np.sum(self.time_series[idx])
                diff = abs(series_sum - target_load)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx
                    best_load = self.time_series[idx]
            return best_idx, best_load

        for idx in remaining_indices:
            new_load = current_load + self.time_series[idx]
            new_sum = np.sum(new_load)

            # Оцінюємо якість розподілу
            non_zero_loads = new_load[new_load > 0]
            if len(non_zero_loads) == 0:
                continue

            variance = np.var(non_zero_loads)
            max_load = np.max(non_zero_loads)
            min_load = np.min(non_zero_loads)
            relative_diff = (max_load - min_load) / max_load

            # Штрафуємо відхилення від цільового навантаження
            target_diff = abs(new_sum - target_load) / target_load

            # Комбінований критерій
            score = variance + relative_diff + target_diff

            if score < best_score:
                best_score = score
                best_idx = idx
                best_load = new_load

        return best_idx, best_load

    def balance_loads(self) -> tuple:
        """Виконує балансування навантаження"""
        groups = []
        remaining_indices = set(range(self.num_series))

        iteration = 0
        while remaining_indices:
            print(f"\nІтерація {iteration}, "
                  f"цільове навантаження: {self.get_target_load(iteration):.2f}")

            current_group = []
            current_load = np.zeros(self.num_slots)

            attempts = 0
            max_attempts = len(remaining_indices)

            while remaining_indices and attempts < max_attempts:
                attempts += 1

                best_idx, new_load = self.find_best_series(
                    current_load, remaining_indices, iteration)

                if best_idx is None:
                    break

                # Перевіряємо чи новий розподіл в межах допустимого відхилення
                if len(current_group) > 0 and not self.check_load_balance(new_load, iteration):
                    print(f"  Перевищено допустиме відхилення")
                    break

                print(f"  Додаємо ряд {best_idx} до групи {iteration+1}")
                current_group.append(best_idx)
                current_load = new_load
                remaining_indices.remove(best_idx)
                attempts = 0

                self.visualize_current_state(groups, current_group,
                                             current_load, iteration)

            if current_group:
                groups.append(current_group)
                print(f"Група {iteration+1} сформована: {current_group}")
                iteration += 1
            else:
                # Якщо не вдалося сформувати групу, збільшуємо допустиме відхилення
                print("Не вдалося сформувати групу, збільшуємо допустиме відхилення")
                if remaining_indices:
                    idx = remaining_indices.pop()
                    groups.append([idx])
                    iteration += 1

        return groups

    def visualize_current_state(self, groups: list, current_group: list,
                                current_load: np.ndarray, iteration: int):
        """Візуалізує поточний стан розподілу"""
        plt.figure(figsize=(12, 6))

        # Показуємо всі сформовані групи
        for i, group in enumerate(groups):
            group_load = np.sum(self.time_series[group], axis=0)
            plt.plot(range(self.num_slots), group_load,
                     label=f'Group {i+1} (series: {group})')

        # Показуємо поточну групу
        plt.plot(range(self.num_slots), current_load,
                 '--', label=f'Group {iteration+1} (in progress: {current_group})')

        # Показуємо цільове навантаження та допустимі межі
        target = self.get_target_load(iteration)
        current_max_deviation = self.max_deviation * (1 + iteration * 0.1)

        if np.sum(current_load) > 0:
            plt.axhline(y=target/self.num_slots, color='r', linestyle=':',
                        label=f'Target load ({target/self.num_slots:.2f})')
            plt.axhline(y=target/self.num_slots * (1 + current_max_deviation),
                        color='r', linestyle='--',
                        label=f'Bounds (±{current_max_deviation*100:.0f}%)')
            plt.axhline(y=target/self.num_slots * (1 - current_max_deviation),
                        color='r', linestyle='--')

        plt.xlabel('Time slot')
        plt.ylabel('Load')
        plt.title(f'Distribution State (Iteration {iteration+1})')
        plt.legend()
        plt.grid(True)
        plt.show()

# Приклад використання
time_series = [
    [0, 0, 3, 8, 5, 0],  # Ряд 0
    [0, 4, 0, 0, 0, 0],  # Ряд 1
    [2, 3, 4, 0, 0, 1],  # Ряд 2
    [0, 0, 0, 5, 4, 3],  # Ряд 3
    [1, 1, 1, 1, 1, 1],  # Ряд 4
    [0, 2, 4, 6, 4, 2],  # Ряд 5
]

balancer = AdaptiveLoadBalancer(time_series, max_deviation=0.3, growth_factor=1.5)
groups = balancer.balance_loads()
print("\nФінальний розподіл груп:", groups)