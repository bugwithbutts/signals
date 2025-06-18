import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

def H_f(f):
    f = np.array(f)
    result = np.zeros_like(f, dtype=float)
    result = np.where(f < 100, 0.0, result)
    result = np.where((f >= 100) & (f <= 300), 2.0, result)
    result = np.where((f >= 300) & (f <= 700), -f / 200 + 7/2, result)
    result = np.where((f >= 700) & (f <= 1200), f / 500 - 7/5, result)
    result = np.where((f >= 1200) & (f <= 1500), 1.0, result)
    result = np.where(f > 1500, 0.0, result)
    return result


# Расчет фильтра
def bandpass_filter(P: int, K: int, F: float):

    omega = np.arange(0.0, np.pi, np.pi / K)

    freq = omega / (2.0 * np.pi / F)

    A_resp_band = H_f(freq)

    M = (P - 1) // 2
    a_matrix = np.asmatrix(A_resp_band[0:K]).T
    F_matrix = np.asmatrix(np.zeros((K, M)))
    M_range = np.arange(M, 0, -1)

    # Заполнение матрицы F
    for i in range(0, K - 1):
        F_matrix[i, 0:M] = 2.0 * np.sin(omega[i + 1] * M_range)

    # Вычисление импульсной характеристики фильтра методом наименьших квадратов
    h_matr = np.linalg.inv(F_matrix.T * F_matrix) * F_matrix.T * a_matrix
    h_half = np.asarray(h_matr.T)[0]

    # Получили только половину импульсной характеристики. Вторая половина импульсной характеристики - развернутая со знаком минус. В середине фильтра - 0
    h = np.zeros(P)
    h[0:M] = h_half
    h[M + 1:P] = -np.flip(h_half)
    return h

F = 8000  # Частота дискретизации
L = 2  # Продолжительность сигнала
N = L * F
coef = 2 * np.pi / F  # Коэфицент преобразования частот
P = 1001  # Порядок фильтра (должен быть нечетный)

bands = np.array([0, 99, 100, 299, 300, 699, 700, 1199, 1200, 1500, 1501, F // 2])

#Сравнение с библиотечным фильтром
bandpass_lib = sc.signal.firls(P, bands, H_f(bands), fs=F)
bandpass_impl = bandpass_filter(P, P, F)

freq = np.linspace(0.0, F / 2.0, N // 2)
resp_impl = sc.signal.freqz(bandpass_impl, 1, freq, fs=F)
resp_lib = sc.signal.freqz(bandpass_lib, 1, freq, fs=F)

# Вывод АЧХ фильтров
plt.xlim(0, 2000)
plt.plot(freq, np.abs(resp_lib[1]))
plt.plot(freq, np.abs(resp_impl[1]))
plt.legend(["Встроенная функция", "Самописная функция"])
plt.show()
