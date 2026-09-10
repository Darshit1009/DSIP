import numpy as np
import matplotlib.pyplot as plt

# Discrete-time signals
x = np.array([1, 2, 3, 2, 1])
y = np.array([0, 1, 2, 1, 0])

# Autocorrelation
Rxx = np.correlate(x, x, mode='full')

# Cross-correlation
Rxy = np.correlate(x, y, mode='full')

# Plot
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.stem(x)
plt.title("Signal x[n]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 2, 2)
plt.stem(y)
plt.title("Signal y[n]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 2, 3)
plt.stem(Rxx)
plt.title("Autocorrelation")
plt.xlabel("Sample")
plt.ylabel("Correlation")
plt.grid()

plt.subplot(2, 2, 4)
plt.stem(Rxy)
plt.title("Cross-correlation")
plt.xlabel("Sample")
plt.ylabel("Correlation")
plt.grid()

plt.tight_layout()
plt.show()