import statsmodels.api as sm
import numpy as np

# Часовий ряд (синтетичні дані)
time_series = np.array([10, 12, 13, 11, 9, 35, 8, 12, 14, 11, 50, 55, 52, 56])

# Декомпозиція часового ряду
decomposition = sm.tsa.seasonal_decompose(time_series, period=4)

# Виділення компонент
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

print("Тренд:", trend)
print("Сезонність:", seasonal)
print("Залишкова складова (аномалії можуть бути тут):", residual)