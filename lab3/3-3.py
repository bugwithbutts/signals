import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin2, freqz

Fc = 30000  
Fs = 44000 
N = 101         

def H_comp(w):
    # Divide by 2 * pi to make angle freq to linear
    return 1 / (1 - (Fs * w) / (Fc * 2 * np.pi))

num_freqs = 1024
w = np.linspace(0, np.pi, num_freqs)

H_desired = H_comp(w)

h = firwin2(N, w/np.pi, H_desired)

plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.plot(h, 'b-', linewidth=2)
plt.title('Импульсная характеристика фильтра')
plt.xlabel('Номер отсчета (n)')
plt.ylabel('Амплитуда h[n]')
plt.grid(True)

plt.subplot(1, 2, 2)

w_filt, H_filt = freqz(h, worN=2048)

plt.plot(w_filt, np.abs(H_filt), 'b-', label='Реальная характеристика')
plt.plot(w, H_desired, 'r--', label='Желаемая характеристика')

plt.title('Частотная характеристика')
plt.xlabel('Цифровая частота (рад/отсчет)')
plt.ylabel('Амплитуда |H(ω)|')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()