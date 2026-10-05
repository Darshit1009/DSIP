import numpy as np
import matplotlib.pyplot as plt

# Input discrete-time signal

signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Compute FFT

fft_result = np.fft.fft(signal)

# Compute magnitude spectrum

magnitude_spectrum = np.abs(fft_result)

# Compute phase spectrum

phase_spectrum = np.angle(fft_result)

# Compute IFFT

reconstructed_signal = np.fft.ifft(fft_result)

# Display results

print("Original Signal:")
print(signal)

print("\nFFT Result:")
print(fft_result)

print("\nMagnitude Spectrum:")
print(magnitude_spectrum)

print("\nPhase Spectrum:")
print(phase_spectrum)

print("\nReconstructed Signal:")
print(reconstructed_signal.real)

# Plot original and reconstructed signal

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(signal, marker='o')
plt.title("Original Signal")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(reconstructed_signal.real, marker='o')
plt.title("Reconstructed Signal using IFFT")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()