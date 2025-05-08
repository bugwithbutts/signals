import numpy as np
import matplotlib.pyplot as plt

def get_fft(signal, fs):
    N = len(signal)
    spec = np.fft.fft(signal)
    amp = 2 * np.abs(spec) / N
    amp[0] /= 2
    freqs = np.fft.fftfreq(N, 1 / fs)
    return freqs, amp

fig, axes = plt.subplots(1, 2)

fs = 1000
t = np.arange(0, 1, 1 / fs)

signal1 = np.hstack((np.exp(-t * t), np.zeros(0)))
signal2 = np.hstack((np.cos(np.pi * t / 2), np.zeros(1000)))

# signal1 = np.blackman(len(signal1)) * signal1
signal2 = np.blackman(len(signal2)) * signal2

x, y = get_fft(signal1, fs)
axes[0].stem(x, y, 'b', markerfmt=" ", basefmt="-b")
axes[0].set_xlim(-10, 10)

x, y = get_fft(signal2, fs)
axes[1].stem(x, y, 'b', markerfmt=" ", basefmt="-b")
axes[1].set_xlim(-10, 10)

plt.tight_layout()
plt.show()