import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, hilbert, remez
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

Fs, x = wav.read("lab3/data files/encoded.wav")
bandpass_coeffs = remez(101, [0, 200, 300, 3000, 3100, Fs/2], [0, 1, 0], fs=Fs)
x = lfilter(bandpass_coeffs, 1.0, x)
wa = 3500 * 2 * np.pi
t = np.arange(0, x.shape[0]) / Fs
b = remez(101, [0, Fs/2], [1],
                    type='hilbert', fs=Fs)
hsig = lfilter(b, 1.0, x)
y = hsig * np.sin(t * wa) - np.cos(t * wa) * x
y = y / np.max(np.abs(y)) * 32767 #???
y = y.astype(np.int16)
wav.write('lab3/data files/decoded.wav', Fs, y)