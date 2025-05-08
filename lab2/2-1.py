import numpy as np
import matplotlib.pyplot as plt

f = 2
w0 = 2 * np.pi * f

fig, axes = plt.subplots(1, 3)
x = np.arange(0, 1, 0.001)
axes[0].plot(x, np.sin(w0 * x))
axes[0].set_ylim(-2, 2)
axes[0].set_title("Original signal")

fs = 5

samples = np.arange(0, 1, 1 / fs)
axes[1].plot(samples, np.sin(w0 * samples))
axes[1].set_ylim(-2, 2)
axes[1].set_title("Samples")

y = 0
for i in samples:
    # np.sinc has already been using pi * x
    y += np.sin(i * w0) * np.sinc(fs * (x - i))

axes[2].plot(x, y)
axes[2].set_ylim(-2, 2)
axes[2].set_title("Reconstructed signal")

plt.tight_layout()
plt.show()

