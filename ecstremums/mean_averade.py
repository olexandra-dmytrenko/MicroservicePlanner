import numpy as np
import pandas as pd

# Часовий ряд
time_series = pd.Series([10, 12, 13, 11, 9, 35, 8, 12, 14, 11])

# Обчислюємо рухоме середнє з вікном 3
moving_avg = time_series.rolling(window=3).mean()
std_dev = np.std(time_series)

# Визначаємо екстремуми
extremes = time_series[(time_series > moving_avg + 2 * std_dev) | (time_series < moving_avg - 2 * std_dev)]

print("Рухоме середнє:", moving_avg)
print("Екстремуми:", extremes)
