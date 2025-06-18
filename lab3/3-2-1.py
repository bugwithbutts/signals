import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

sampling, signal = wav.read("lab3/data files/tune.wav") # частота дискретизации #  сигнала
fig, axes = plt.subplots(1, 2)
# Параметры фильтра
N = 101  # Количество коэффициентов фильтра (порядок + 1)
cutoff_freq = 0.04  # Частота среза

# Создание КИХ-фильтра с использованием оконной функции Блэкмана
b = firwin(N, cutoff_freq, window='blackman')

# Применение фильтра к сигналу
filtered = lfilter(b, 1.0, signal).astype(np.int16)

offset = sampling
size = sampling // 25
axes[0].plot(signal[offset:offset+size])
axes[0].plot(filtered[offset:offset+size])
axes[0].set_xlabel('Время')
axes[0].set_ylabel('Сигнал')
axes[0].set_title('Сигнал')

# Расчет амплитудного спектра
frq = np.fft.fftfreq(len(filtered), 1/sampling)
fft1 = np.fft.fft(filtered)
fft2 = np.fft.fft(signal)
axes[1].plot(frq, np.abs(fft2))
axes[1].plot(frq, np.abs(fft1))
axes[1].set_xlabel('Частота')
axes[1].set_ylabel('Амплитуда')
axes[1].set_title('Спектр отфильтрованного сигнала')
plt.show()