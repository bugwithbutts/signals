import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

sampling, signal = wav.read('lab2\\data files\\test5.wav')
y = np.array(np.fft.fft(signal))
print(len(y))
N = (len(y) - 1) // 2
N //= 4
zero_val = y[0]
C = y[1:N]
B = y[N:2*N]
D = y[2*N:3*N]
A = y[3*N:4*N]

# [::-1] to reverse
# zero_val has no conj pair
decode = np.hstack((zero_val, A, B, C, D, np.conj(D)[::-1], np.conj(C)[::-1], np.conj(B)[::-1], np.conj(A)[::-1]))


filtered_signal = np.real(np.fft.ifft(decode)).astype(np.int16)

wav.write('lab2\\data files\\decode.wav', sampling, filtered_signal)

