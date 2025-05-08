import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

sampling, signal = wav.read('lab2\\data files\\test5.wav')
y = np.array(np.fft.fft(signal))
N = len(y)
print(N)
N //= 4
C = y[0:N]
B = y[N:2*N]
D = y[2*N:3*N]
A = y[3*N:]
decode = np.hstack((A, B, C, D))


filtered_signal = np.real(np.fft.ifft(decode)).astype(np.int16)

wav.write('lab2\\data files\\decode.wav', sampling, filtered_signal)

