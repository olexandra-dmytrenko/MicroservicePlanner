import numpy as np
import matplotlib.pyplot as plt

# Часовий ряд
time_series = np.array([10, 12, 13, 31, 9, 35, 8, 12, 14, 38, 32, 50, 55, 52, 56, 52, 49, 40, 34, 30, 25, 20, 22, 11, 14])

# time_series = np.array([10, 12, 13, 11, 9, 35, 8, 12, 14, 11])

# Обчислюємо середнє та стандартне відхилення
mean = np.mean(time_series)
std_dev = np.std(time_series)

# Визначаємо екстремуми (аномалії)
threshold = 1  # поріг у стандартних відхиленнях
extremes = time_series[(time_series > mean + threshold * std_dev)] #| (time_series < mean - threshold * std_dev)]

# Знаходимо індекси першого та останнього елементів в підмножині extremes
first_idx = np.where(time_series == extremes[0])[0][0]
last_idx = np.where(time_series == extremes[-1])[0][0]

# Ініціалізуємо масив для розширення extremes
extremes_extention_left = []

# Перевірка елементів ліворуч від першого елементу extremes
for i in range(first_idx - 1, -1, -1):
    if time_series[i] > mean:
        extremes_extention_left.append(time_series[i])
    else:
        break

# Перевірка елементів праворуч від останнього елементу extremes
extremes_extention_right = []
for i in range(last_idx + 1, len(time_series)):
    if time_series[i] > mean:
        extremes_extention_right.append(time_series[i])
    else:
        break

# Знаходимо максимальне значення за виключенням extremes та extremes_extention
mask = ~np.isin(time_series, np.concatenate([extremes_extention_left, extremes, extremes_extention_right]))
new_max_value = np.max(time_series[mask])

#print("extremes_extention:", extremes_extention)
print("new_max_value:", new_max_value)


def extract_smaller_values_then(extremes_extention, new_max_value):
    if extremes_extention:
        min_extention = np.min(extremes_extention)
        if new_max_value > min_extention:
            extremes_extention = [x for x in extremes_extention if x >= new_max_value]
    return extremes_extention


# Якщо нове максимальне значення більше за мінімальне значення у extremes_extention, видаляємо всі менші значення
extremes_extention_left = extract_smaller_values_then(extremes_extention_left, new_max_value)
extremes_extention_right = extract_smaller_values_then(extremes_extention_right, new_max_value)

print("extremes_extention:", extremes_extention_left)
print("extremes_extention:", extremes_extention_right)

# Об'єднуємо extremes та extremes_extention
final_extremes = np.concatenate([extremes_extention_left, extremes, extremes_extention_right])

print("Final extremes:", final_extremes)
print("Середнє:", mean)
print("Стандартне відхилення:", std_dev)
print("Екстремуми:", extremes)
print("Екстремуми розширені:", final_extremes)

plt.plot(time_series, label='Часовий ряд')
plt.axvline(np.where(time_series == extremes[0])[0][0], color='g', linestyle='dotted', label='Початок екстремуму')
plt.axvline(np.where(time_series == final_extremes[0])[0][0], color='b', linestyle='dashed', label='Початок розширеного екстремуму')
plt.axvline(np.where(time_series == extremes[-1])[0][-1], color='g', linestyle='dotted', label='Кінець екстремуму')
plt.axvline(np.where(time_series == final_extremes[-1])[0][-1], color='b', linestyle='dashed', label='Кінець розширеного екстремуму')
plt.axhline(y=new_max_value, color='r', linestyle='--', label=f'Лінія поділу = {new_max_value} од.')
plt.axhline(y=mean, color='c', linestyle='dashdot', label=f'Середнє =  {mean:.2f} од.')
plt.axhline(y=mean + std_dev, color='y', linestyle='dotted', label=f'Середнє + стандартне відхилення = {mean + std_dev:.2f} од.')
plt.axhline(y=mean - std_dev, color='y', linestyle='dotted', label=f'Середнє - стандартне відхилення = {mean - std_dev:.2f} од.')
plt.title('Виділення часової рамки та навантаження для створення окремого екземпляра мікросервісу', loc='center')
plt.xlabel('Години')
plt.ylabel('Навантаження, од.')
plt.legend()
plt.show()

#-----------Розділення графіків------------#
# 1. Графік зі значеннями меншими та рівними за new_max_value
modified_series = np.where(time_series > new_max_value, new_max_value, time_series)

# 2. Графік зі значеннями, більшими за new_max_value, з відніманням new_max_value
above_new_max = np.where(time_series > new_max_value, time_series - new_max_value, 0)

# Налаштування графіків
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Графік 1
axs[0].plot(modified_series, label='Навантаження без піка', color='b')
axs[0].set_title('Навантаження ≤ за максимальне дозволене (35 од.)')
axs[0].set_ylabel('Навантаження стандартне, од.')
axs[0].axhline(y=new_max_value, color='r', linestyle='--', label='Обмеження максимумом = 35 од.')
axs[0].legend()
axs[0].grid()

# Графік 2
axs[1].plot(above_new_max, label='Навантаження > за максимальне дозволене (35 од.)', color='g')
axs[1].set_title('Пікове навантаження, винесене в окремий сервіс')
axs[1].set_ylabel('Навантаження підвищене, од.')
axs[1].legend()
axs[1].grid()

# Налаштування осі X
axs[1].set_xticks(np.arange(0, len(time_series), step=1))
axs[1].set_xticklabels(np.arange(0, 25, step=1))

# Показати графіки
plt.xlabel('Години')
plt.tight_layout()
plt.show()