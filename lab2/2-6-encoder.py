import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

sampling, signal = wav.read('lab2\\data files\\filtered.wav')
message = "N01010101"
N = 20
step = len(signal) // N

dft = np.fft.fft(signal[0:step])
ph = np.angle(dft)
lst = ph
delta = []
for i in range(step, len(signal), step):
    dft = np.fft.fft(signal[i:i+step])
    tmp = np.angle(dft)
    delta.append(tmp - lst)
    lst = tmp

half = len(ph) // 2
# Dont touch X[0]????
for i in range(0, len(message) + 1): #???
    if i < len(message) and message[i] == '1':
        ph[i] = -np.pi / 2
    elif i < len(message):
        ph[i] = np.pi / 2
    else:
        ph[i] = 0

for i in range(1, half):
    ph[half * 2 - i] = -ph[i]

lst = ph
dft = np.abs(np.fft.fft(signal[0:step])) * np.exp(1j * ph)
filt = np.real(np.fft.ifft(dft)).astype(np.int16)
ans = [np.array(filt)]
for i in range(step, len(signal), step):
    lst += delta[i // step - 1]
    dft = np.abs(np.fft.fft(signal[i:i+step])) * np.exp(1j * lst)
    filt = np.real(np.fft.ifft(dft)).astype(np.int16)
    ans.append(np.array(filt))

ans = np.hstack(ans)
wav.write('lab2\\data files\\encoded.wav', sampling, ans)
