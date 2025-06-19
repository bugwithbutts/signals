import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Загрузка данных
mixed_signal = np.loadtxt('lab3/data files/ecg fetal + mother.txt')
mother_ecg = np.loadtxt('lab3/data files/ecg mother.txt')
true_fetal_ecg = np.loadtxt('lab3/data files/ecg fetal.txt')

# Параметры адаптивного фильтра
filter_order = 128  # Порядок фильтра
mu = 0.01  # Шаг адаптации

# Инициализация адаптивного фильтра (LMS алгоритм)
w = np.zeros(filter_order)
estimated_maternal = np.zeros_like(mixed_signal)
fetal_ecg_estimated = np.zeros_like(mixed_signal)

# Адаптивная фильтрация
for n in range(filter_order, len(mixed_signal)):
    x_window = mother_ecg[max(0, n-filter_order):n]
    if len(x_window) != filter_order:
        x_window = np.pad(x_window, (0, filter_order-len(x_window)))
    estimated_maternal[n] = np.dot(w, x_window)
    fetal_ecg_estimated[n] = mixed_signal[n] - estimated_maternal[n]
    w = w + mu * fetal_ecg_estimated[n] * x_window

# Масштабирование сигнала (учет ослабления в 2 раза)
fetal_ecg_estimated = fetal_ecg_estimated * 2

# Визуализация результатов
plt.figure(figsize=(12, 8))
plt.subplot(3,1,1)
plt.plot(mixed_signal[:3000], label='Смешанный сигнал')
plt.legend()
plt.subplot(3,1,2)
plt.plot(fetal_ecg_estimated[:3000], label='Выделенный ЭКГ плода')
plt.legend()
plt.subplot(3,1,3)
plt.plot(true_fetal_ecg[:3000], label='Истинный ЭКГ плода')
plt.legend()
plt.tight_layout()
plt.show()

# Оценка качества
mse = mean_squared_error(true_fetal_ecg, fetal_ecg_estimated[:len(true_fetal_ecg)])
print(f"Среднеквадратичная ошибка: {mse:.4f}")
correlation = np.corrcoef(true_fetal_ecg, fetal_ecg_estimated[:len(true_fetal_ecg)])[0,1]
print(f"Коэффициент корреляции: {correlation:.4f}")