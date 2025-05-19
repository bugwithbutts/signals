import numpy as np
import matplotlib.pyplot as plt

def get_fft(signal, fs):
    N = len(signal)
    spec = np.fft.fft(signal)
    amp = np.abs(spec) * np.sqrt(1/(2*np.pi)) / fs
    amp = np.fft.fftshift(amp)
    freqs = np.fft.fftfreq(N, 1 / fs)
    freqs = np.fft.fftshift(freqs)
    return freqs, amp

fig, axes = plt.subplots(1, 2)

fs = 30000
t = np.arange(-1, 1, 1 / fs)

signal1 = np.exp(-t * t)
signal2 = np.cos(np.pi * t / 2)

x, y = get_fft(signal1, fs)
axes[0].plot(x, y)
axes[0].set_xlim(-10, 10)   
teor1 = np.abs((1 / np.sqrt(2)) * np.exp(-((np.pi*2*x)**2)/4))
axes[0].plot(x, teor1)

x, y = get_fft(signal2, fs)
axes[1].plot(x, y)
axes[1].set_xlim(-10, 10)
teor2 = np.abs(np.sqrt(2/np.pi)*0.5*(np.sinc(2*x-1/2)+np.sinc(2*x+1/2)))
axes[1].plot(x, teor2)
plt.tight_layout()
plt.show()