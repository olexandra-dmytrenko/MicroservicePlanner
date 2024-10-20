import numpy as np

# Часовий ряд (синтетичні дані)
time_series = np.array([10, 12, 13, 11, 9, 35, 8, 12, 14, 11, 50, 55, 52, 56])

# Обчислення першої та другої похідних
first_derivative = np.diff(time_series)
second_derivative = np.diff(first_derivative)

# Визначення початку та кінця екстремуму
extremum_start = np.where(second_derivative < 0)[0][0]
extremum_end = np.where(second_derivative > 0)[0][-1]

print("Початок екстремуму:", extremum_start)
print("Кінець екстремуму:", extremum_end)