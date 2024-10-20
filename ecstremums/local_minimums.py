import numpy as np
from scipy.signal import argrelextrema

# Часовий ряд
time_series = np.array([10, 12, 13, 11, 9, 35, 8, 12, 14, 11, 50, 55, 52, 56])

# Пошук локальних максимумів
local_maxima = argrelextrema(time_series, np.greater)
local_minima = argrelextrema(time_series, np.less)

print("Локальні максимуми:", local_maxima)
print("Локальні мінімуми:", local_minima)