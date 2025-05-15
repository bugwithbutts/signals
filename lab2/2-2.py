import numpy as np
import matplotlib.pyplot as plt

def get_fft(signal, fs):
    N = len(signal)
    spec = np.fft.fft(signal)
    amp = np.abs(spec) / fs
    amp = np.fft.fftshift(amp)
    freqs = np.fft.fftfreq(N, 1 / fs)
    freqs = np.fft.fftshift(freqs)
    return freqs, amp

fig, axes = plt.subplots(1, 2)

fs = 1000
t = np.arange(0, 1, 1 / fs)

signal1 = np.exp(-t * t)
signal2 = np.cos(np.pi * t / 2)

x, y = get_fft(signal1, fs)
axes[0].plot(x, y)
axes[0].set_xlim(-10, 10)
teor1 = np.sqrt(np.pi) * np.exp(-x**2/4)   
axes[0].plot(x, teor1)

x, y = get_fft(signal2, fs)
axes[1].plot(x, y)
axes[1].set_xlim(-10, 10)
teor2 = 4 * np.pi * np.cos(x) / (np.pi ** 2 - 4 * x ** 2)
axes[1].plot(x, teor2)
plt.tight_layout()
plt.show()