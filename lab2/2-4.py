import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_fft(signal, fs):
    N = len(signal)
    spec = np.fft.fft(signal)
    amp = np.abs(spec)
    freqs = np.fft.fftfreq(N, 1 / fs)
    return freqs, amp

fig, axes = plt.subplots(2, 2)
time, signal = [], []
with open("lab2\\data files\\ecg.dat") as file:
    for line in file.readlines():
        vals = line.split()
        time.append(float(vals[0]))
        signal.append(float(vals[1]))

T = time[2] - time[1]
axes[0, 0].set_title("Original ECG")
axes[0, 0].plot(time[:4000], signal[:4000])

freq, amp = get_fft(signal, 1 / T)
axes[0, 1].plot(freq, amp)
axes[0, 1].set_title("Original spectrum")
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].set_xlabel('Frequency')

filter = np.array([0 if abs(f - 50) < 2 or abs(f + 50) < 2 else 1 for f in freq])
filtered = filter * np.fft.fft(signal)

axes[1, 1].plot(freq, np.abs(filtered))
axes[1, 1].set_title("Filtered spectrum")
axes[1, 1].set_ylabel('Amplitude')
axes[1, 1].set_xlabel('Frequency')
axes[0, 1].set_xlim(-100, 100)

filtered_signal = np.real(np.fft.ifft(filtered))
axes[1, 0].set_title("Filtered ECG")
axes[1, 0].plot(time[:4000], filtered_signal[:4000])
plt.tight_layout()
plt.show()