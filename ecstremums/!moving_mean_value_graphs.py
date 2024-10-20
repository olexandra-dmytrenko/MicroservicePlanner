import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Часовий ряд (синтетичні дані)
time_series = pd.Series([10, 12, 13, 11, 9, 35, 8, 12, 14, 11, 50, 55, 52, 56])

# Рухоме середнє з вікном 3
moving_avg = time_series.rolling(window=3).mean()

# Пошук точок екстремуму
threshold = 10  # поріг для екстремуму
extremum_start = np.where(time_series > moving_avg + threshold)[0][0]
extremum_end = np.where(time_series > moving_avg + threshold)[0][-1]

# Виведення результатів
print("Початок екстремуму:", extremum_start)
print("Кінець екстремуму:", extremum_end)

# Візуалізація
plt.plot(time_series, label='Часовий ряд')
plt.plot(moving_avg, label='Рухоме середнє')
plt.axvline(extremum_start, color='r', linestyle='--', label='Початок екстремуму')
plt.axvline(extremum_end, color='g', linestyle='--', label='Кінець екстремуму')
plt.legend()
plt.show()
