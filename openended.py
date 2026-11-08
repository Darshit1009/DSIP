import numpy as np
import matplotlib.pyplot as plt

# --- Signal 1: Discrete sequence x[n] ---
# Defined at specific integers n
n1 = np.arange(-3, 7)
x1 = np.array([2, 3, 2, 2, 2, 3, 2, 3, 2, 3]) 
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.stem(n1, x1)
plt.title("Signal 1: Discrete Sequence")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)

# --- Signal 2: Staircase sequence x[n] ---
# A discrete-time signal where values remain constant over intervals
n2 = np.arange(0, 7)
x2 = np.array([1, 1, 2, 3, 2, 1, 1])
plt.subplot(1, 3, 2)
plt.stem(n2, x2) # Discrete representation
plt.title("Signal 2: Staircase")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)

# --- Signal 3: Piecewise discrete sequence x[n] ---
# Evaluating the linear ramp at discrete points n
n3 = np.arange(-2, 3)
x3 = np.array([-2, -2, 0, 2, 2]) # Discrete values at n = -2, -1, 0, 1, 2
plt.subplot(1, 3, 3)
plt.stem(n3, x3)
plt.title("Signal 3: Piecewise")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid(True)

plt.tight_layout()
plt.show()

n = np.arange(-5, 15)

# --- Signal 4: x[n] = u(n) - u(n-3) - 5*u(n-7) ---
# Unit step function u(n)
u = lambda n: np.heaviside(n, 1)
x4 = u(n) - u(n-3) - 5*u(n-7)

# --- Signal 5: x[n] = delta(n) + 3*delta(n-1) + 5*delta(n+1) ---
# Unit impulse function delta(n)
delta = lambda n: np.where(n == 0, 1, 0)
x5 = delta(n) + 3*delta(n-1) + 5*delta(n+1)

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.stem(n, x4)
plt.title("Signal 4: x[n] = u(n) - u(n-3) - 5u(n-7)")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.stem(n, x5)
plt.title("Signal 5: x[n] = delta(n) + 3*delta(n-1) + 5*delta(n+1)")
plt.grid(True)

plt.tight_layout()
plt.show()