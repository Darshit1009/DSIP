import numpy as np
import matplotlib.pyplot as plt

# Discrete time index
n = np.arange(-10, 11)

# Unit Impulse Sequence
impulse = np.where(n == 0, 1, 0)

# Unit Step Sequence
step = np.where(n >= 0, 1, 0)

# Ramp Sequence
ramp = np.where(n >= 0, n, 0)

# Exponential Sequence
a = 0.8
exponential = np.where(n >= 0, a**n, 0)

# Sinusoidal Sequence
A = 1
omega = np.pi / 4
sinusoid = A * np.sin(omega * n)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.stem(n, impulse)
plt.title("Unit Impulse Sequence")
plt.xlabel("n")
plt.ylabel("δ(n)")
plt.grid(True)

plt.subplot(3, 2, 2)
plt.stem(n, step)
plt.title("Unit Step Sequence")
plt.xlabel("n")
plt.ylabel("u(n)")
plt.grid(True)

plt.subplot(3, 2, 3)
plt.stem(n, ramp)
plt.title("Ramp Sequence")
plt.xlabel("n")
plt.ylabel("r(n)")
plt.grid(True)

plt.subplot(3, 2, 4)
plt.stem(n, exponential)
plt.title("Exponential Sequence")
plt.xlabel("n")
plt.ylabel("x(n)")
plt.grid(True)

plt.subplot(3, 2, 5)
plt.stem(n, sinusoid)
plt.title("Sinusoidal Sequence")
plt.xlabel("n")
plt.ylabel("x(n)")
plt.grid(True)

plt.tight_layout()
plt.show()