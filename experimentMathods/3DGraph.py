import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Генеруємо дані
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

data = np.array([
    [0.8, 0.6, 0.7],  # Microservice 1
    [0.5, 0.4, 0.5],  # Microservice 2
    [0.9, 0.8, 0.9],  # Microservice 3
    # Add more microservices as needed
])
#plt.scatter(data[:, 0], data[:, 1], data[:, 2])
print(" x = " + str(data[:, 0]) + " y = " + str(data[:, 1]) + " z = "+  str(data[:, 2]))
# Створюємо 3D графік
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Додаємо точки
#ax.scatter(x, y, z, c='r', marker='o')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', marker='o')

# Підписуємо осі
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Показуємо графік
plt.show()
