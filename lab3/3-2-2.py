import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

time, signal = [], []
with open("lab3/data files/ecg.dat") as file:
    for line in file.readlines():
        vals = line.split()
        time.append(float(vals[0]))
        signal.append(float(vals[1]))

sampling_period = time[1]
fig, axes = plt.subplots(1, 3)

axes[0].set_title("Зашумленная ЭКГ")
axes[0].plot(time[:4000], signal[:4000])

N = 101  # Количество коэффициентов фильтра (порядок + 1)
cutoff_freqs = [0.05, 0.14]
# Создание КИХ-фильтра с использованием оконной функции Блэкмана
b = firwin(N, cutoff_freqs, pass_zero='bandstop', window='blackman')
filtered = lfilter(b, 1.0, signal)

axes[1].set_title("Очищенная ЭКГ")
axes[1].plot(time[:4000], filtered[:4000])

# Расчет амплитудного спектра
fft1 = np.fft.fft(filtered)
frq = np.fft.fftfreq(len(filtered), sampling_period)
fft2 = np.fft.fft(signal)
axes[2].plot(frq, np.abs(fft2))
axes[2].plot(frq, np.abs(fft1))
axes[2].set_xlabel('Частота')
axes[2].set_ylabel('Амплитуда')
axes[2].set_title('Спектр отфильтрованного сигнала')
plt.show()