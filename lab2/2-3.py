import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

sampling, signal = wav.read('lab2\\data files\\tune.wav')
n = signal.shape[0]

fig, axes = plt.subplots(1, 2)

spectrum = np.fft.fft(signal)

amplitude = np.abs(spectrum)
frequency = np.fft.fftfreq(n, 1 / sampling)
axes[0].plot(frequency, amplitude)
axes[0].set_title("tune.wav")
axes[0].set_ylabel('Amplitude')
axes[0].set_xlabel('Frequency')
    
filtered_spectrum = np.array(spectrum)
fraction = 30
filtered_spectrum[n//fraction:-n//fraction] = 0

axes[1].plot(frequency, np.abs(filtered_spectrum))
axes[1].set_title("filtered.wav")
axes[1].set_ylabel('Amplitude')
axes[1].set_xlabel('Frequency')
plt.show()

filtered_signal = np.real(np.fft.ifft(filtered_spectrum)).astype(np.int16)

wav.write('lab2\\data files\\filtered.wav', sampling, filtered_signal)