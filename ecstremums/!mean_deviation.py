import numpy as np
import matplotlib.pyplot as plt

# Часовий ряд
time_series = np.array([10, 12, 13, 31, 9, 35, 8, 12, 14, 11, 50, 55, 52, 56, 52, 49, 40, 34, 30, 25, 20, 22, 11, 14])

# time_series = np.array([10, 12, 13, 11, 9, 35, 8, 12, 14, 11])

# Обчислюємо середнє та стандартне відхилення
mean = np.mean(time_series)
std_dev = np.std(time_series)

# Визначаємо екстремуми (аномалії)
threshold = 1  # поріг у стандартних відхиленнях
extremes = time_series[(time_series > mean + threshold * std_dev)] #| (time_series < mean - threshold * std_dev)]

# Create a mask to exclude elements that are in extremes
mask = ~np.isin(time_series, extremes)

# Filter the time_series array and find the maximum
new_max_value = np.max(time_series[mask])

print("Середнє:", mean)
print("Стандартне відхилення:", std_dev)
print("Екстремуми:", extremes)

plt.plot(time_series, label='Часовий ряд')
plt.axvline(np.where(time_series == extremes[0])[0][0], color='g', linestyle='--', label='Початок екстремуму')
plt.axvline(np.where(time_series == extremes[-1])[0][-1], color='g', linestyle='--', label='Кінець екстремуму')
plt.axhline(y=new_max_value, color='r', linestyle='--', label=f'Лінія поділу = {new_max_value}')
plt.axhline(y=mean, color='c', linestyle='--', label=f'Середнє =  {mean:.2f}')
plt.axhline(y=mean + std_dev, color='y', linestyle='--', label=f'Середнє + cтандартне відхилання = {mean + std_dev:.2f}')
plt.axhline(y=mean - std_dev, color='y', linestyle='--', label=f'Середнє - cтандартне відхилання = {mean - std_dev:.2f}')
plt.title('Виділення часової рамки та навантаження для створення окремого екземпляра мікросервіса', loc='center')
plt.legend()
plt.show()