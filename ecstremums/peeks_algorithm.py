import numpy as np
from scipy.signal import find_peaks

# Часовий ряд (синтетичні дані)
time_series = np.array([10, 12, 13, 11, 9, 35, 8, 12, 14, 11, 50, 55, 52, 56])

# Виявлення піків (екстремумів)
peaks, _ = find_peaks(time_series, height=10)

# Виведення індексів піків
print("Індекси піків:", peaks)