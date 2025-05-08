import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

sampling, signal = wav.read('lab2\\data files\\encoded.wav')
N = 100
step = len(signal) // N
dft = np.fft.fft(signal[0:step])
ph = np.angle(dft)
lst = ph
message = ""
for i in range(1, len(ph) // 2):
    if abs(ph[i] - np.pi / 2) < 0.5:
        message += '1'
    elif abs(ph[i] - -np.pi / 2) < 0.5:
        message += '0'
    else:
        break
print(message)
