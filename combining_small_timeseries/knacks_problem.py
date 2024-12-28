import numpy as np
import matplotlib.pyplot as plt

def explain_knapsack_balancing(time_series: list, target_load: float = None):
    """
    Демонстраційна версія алгоритму балансування на основі задачі пакування рюкзака
    """
    time_series = np.array(time_series)
    num_series = len(time_series)
    num_slots = len(time_series[0])

    # Якщо цільове навантаження не задане, обчислюємо його як середнє
    if target_load is None:
        target_load = np.mean([np.sum(series) for series in time_series])

    print(f"Цільове навантаження: {target_load:.2f}")

    groups = []
    remaining_indices = set(range(num_series))

    def visualize_current_state(current_load=None):
        """Візуалізує поточний стан розподілу"""
        plt.figure(figsize=(12, 6))

        # Показуємо всі сформовані групи
        for i, group in enumerate(groups):
            group_load = np.sum(time_series[group], axis=0)
            plt.plot(range(num_slots), group_load,
                     label=f'Group {i+1} (series: {group})')

        # Показуємо поточну групу, яка формується
        if current_load is not None:
            plt.plot(range(num_slots), current_load,
                     '--', label='Current group (in progress)')

        # Показуємо цільове навантаження
        plt.axhline(y=target_load/num_slots, color='r',
                    linestyle=':', label='Target load per slot')

        plt.xlabel('Time slot')
        plt.ylabel('Load')
        plt.title('Current Distribution State')
        plt.legend()
        plt.grid(True)
        plt.show()

    group_number = 1
    while remaining_indices:
        print(f"\nФормування групи {group_number}:")
        current_group = []
        current_load = np.zeros(num_slots)

        while remaining_indices:
            best_idx = None
            best_diff = float('inf')

            # Знаходимо найкращий часовий ряд для додавання
            for idx in remaining_indices:
                new_load = current_load + time_series[idx]
                diff = np.mean(np.abs(new_load - target_load/num_slots))

                print(f"  Перевірка ряду {idx}: "
                      f"різниця = {diff:.2f} "
                      f"(поточна найкраща = {best_diff:.2f})")

                if diff < best_diff:
                    best_diff = diff
                    best_idx = idx

            # Перевіряємо чи варто додавати
            new_load = current_load + time_series[best_idx]
            if np.mean(new_load) > target_load * 1.2:
                print(f"  Перевищено допустиме навантаження, "
                      f"починаємо нову групу")
                break

            print(f"  Додаємо ряд {best_idx} до групи {group_number}")
            current_group.append(best_idx)
            current_load = new_load
            remaining_indices.remove(best_idx)

            visualize_current_state(current_load)

        if current_group:
            groups.append(current_group)
            print(f"Група {group_number} сформована: {current_group}")
            group_number += 1

    return groups

# Приклад використання
time_series = [
    [0, 0, 3, 8, 5, 0],  # Ряд 0
    [0, 4, 0, 0, 0, 0],  # Ряд 1
    [2, 3, 4, 0, 0, 1],  # Ряд 2
    [0, 0, 0, 5, 4, 3],  # Ряд 3
    [1, 1, 1, 1, 1, 1],  # Ряд 4
    [0, 2, 4, 6, 4, 2],  # Ряд 5
]

groups = explain_knapsack_balancing(time_series)
print("\nФінальний розподіл груп:", groups)