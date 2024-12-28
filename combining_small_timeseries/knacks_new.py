import numpy as np
import matplotlib.pyplot as plt

def optimize_sparse_timeseries(data, max_target_increase=0.3, verbose=False):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    n_rows, n_cols = data.shape

    # Знаходимо максимальне значення та встановлюємо обмежене цільове
    max_value = np.max(data)
    target_value = max_value * (1 + max_target_increase)

    # Сортуємо рядки за максимальними значеннями
    row_max_values = np.max(data, axis=1)
    sorted_indices = np.argsort(-row_max_values)
    sorted_data = data[sorted_indices]

    # Ініціалізуємо результат та групи
    result = np.zeros_like(data)
    current_sums = np.zeros(n_cols)
    groups = []  # Для зберігання індексів рядків у кожній групі
    current_group = []

    def evaluate_difference(row, current_sums, target):
        potential_sums = current_sums + row
        weighted_diff = np.sum(np.abs(potential_sums - target))
        return weighted_diff

    # Проходимо по відсортованих рядках
    for i in range(n_rows):
        row = sorted_data[i]

        # Перевіряємо, чи покращить додавання цього рядка результат
        current_diff = evaluate_difference(np.zeros_like(row), current_sums, target_value)
        new_diff = evaluate_difference(row, current_sums, target_value)

        if new_diff <= current_diff:
            result[sorted_indices[i]] = row
            current_sums += row
            current_group.append(sorted_indices[i])
        else:
            if current_group:  # Якщо поточна група не порожня
                groups.append(current_group.copy())
                current_group = []

            # Починаємо нову групу
            current_group.append(sorted_indices[i])
            result[sorted_indices[i]] = row
            current_sums = row.copy()

        if verbose:
            print(f"Iteration {i+1}:")
            print(f"Current sums: {current_sums}")
            print(f"Target value: {target_value}")

    # Додаємо останню групу, якщо вона не порожня
    if current_group:
        groups.append(current_group)

    return result, current_sums, target_value, groups

def plot_groups(data, groups):
    plt.figure(figsize=(12, 6))

    # Створюємо кольорову палітру для груп
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(groups)))

    for group_idx, group in enumerate(groups):
        for row_idx in group:
            # Отримуємо індекси ненульових елементів
            non_zero_indices = np.nonzero(data[row_idx])[0]
            values = data[row_idx][non_zero_indices]

            # Малюємо лінію для кожного ряду в групі
            plt.plot(non_zero_indices, values, 'o-',
                     color=colors[group_idx],
                     label=f'Group {group_idx + 1}' if row_idx == group[0] else "")

    plt.grid(True)
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.title('Grouped Time Series')
    plt.legend()
    plt.show()

# Тестові дані та виконання
if __name__ == "__main__":
    test_data = np.array([
        [100, 0, 50, 0, 75, 0],
        [0, 80, 0, 60, 0, 90],
        [30, 0, 40, 0, 20, 0],
        [0, 25, 0, 35, 0, 45],
        [15, 0, 25, 0, 30, 0],
        [0, 10, 0, 20, 0, 15]
    ])

    result, final_sums, final_target, groups = optimize_sparse_timeseries(test_data,
                                                                          max_target_increase=0.3,
                                                                          verbose=True)

    print("\nFinal result:")
    print(result)
    print("\nFinal column sums:")
    print(final_sums)
    print("\nFinal target value:")
    print(final_target)
    print("\nGroups:")
    for i, group in enumerate(groups):
        print(f"Group {i + 1}: {group}")

    # Візуалізуємо результат
    plot_groups(test_data, groups)