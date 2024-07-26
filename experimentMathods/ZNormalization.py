import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt

# Генерація синтетичних даних (годинні дані протягом дня для двох днів)
day1 = np.array([10, 8,  15, 21, 25, 17, 13, 15, 11,  7, 12,  8,  7, 6,   5, 10, 12, 13, 22, 25, 15, 12, 14, 11])
day2 = np.array([20, 24, 30, 40, 50, 36, 26, 24, 22, 20, 18, 16, 14, 12, 10, 20, 24, 30, 40, 50, 36, 26, 24, 22])

# Нормалізація за допомогою Z-оцінки
day1_z = (day1 - np.mean(day1)) / np.std(day1)
day2_z = (day2 - np.mean(day2)) / np.std(day2)
print("McServ 1: " + str(day1_z))
print("McServ 2: " + str(day2_z))

# Перетворення у вертикальні масиви (2D масиви)
day1_z = day1_z.reshape(-1, 1)
day2_z = day2_z.reshape(-1, 1)

# Обчислення DTW відстані між нормалізованими рядами
distance, path = fastdtw(day1_z, day2_z, dist=euclidean)

print(f"DTW distance: {distance}")

# Візуалізація нормалізованих рядів
plt.figure(figsize=(10, 6))
plt.plot(day1_z, label='McServ 1: (Z-normalized)')
plt.plot(day2_z, label='McServ 1: (Z-normalized)')
plt.legend()
plt.title("Normalized Daily Patterns")
plt.xlabel("Hour of Day")
plt.ylabel("Normalized Value")
plt.show()
