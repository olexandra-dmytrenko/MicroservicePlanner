def balance_loads(time_series, num_groups):
    # Перетворюємо часові ряди в суми навантажень
    loads = [sum(series) for series in time_series]

    # Сортуємо навантаження за спаданням
    sorted_loads = sorted(enumerate(loads), key=lambda x: x[1], reverse=True)

    # Створюємо групи
    groups = [[] for _ in range(num_groups)]
    group_sums = [0] * num_groups

    # Розподіляємо навантаження
    for idx, load in sorted_loads:
        # Знаходимо групу з найменшою сумою
        min_sum_group = min(range(num_groups), key=lambda i: group_sums[i])

        # Додаємо навантаження до цієї групи
        groups[min_sum_group].append(idx)
        group_sums[min_sum_group] += load

    return groups, group_sums

# Приклад використання
time_series = [
    [1, 2, 3, 4],  # Сума = 10
    [2, 4, 6, 8],  # Сума = 20
    [3, 3, 3, 3],  # Сума = 12
    [5, 5, 5, 5],  # Сума = 20
]

groups, group_sums = balance_loads(time_series, 2)
print("Розподіл індексів часових рядів по групах:", groups)
print("Сума навантажень в кожній групі:", group_sums)