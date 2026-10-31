
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, freqz, lfilter

# Generation, addition, and filtering of discrete sine waves

# Parameters
fs = 16000                 # Sampling frequency in Hz
duration = 0.01             # Signal duration in seconds
filter_order = 4
ripple = 0.5                # Chebyshev Type-I ripple in dB

frequencies = [1000, 2000, 3000, 4000]

# Generate discrete-time samples
t = np.arange(0, duration, 1 / fs)
N = len(t)

# ------------------------------------------------------------
# TASK 1: Generate and plot four discrete sine waves
# ------------------------------------------------------------

signals = {}

for f in frequencies:
    signals[f] = np.sin(2 * np.pi * f * t)

fig, axes = plt.subplots(4, 1, figsize=(11, 10))

for ax, f in zip(axes, frequencies):
    ax.stem(t * 1000, signals[f], basefmt="k-")
    ax.set_title(f"Discrete Sine Wave - {f // 1000} kHz")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Helper function: Plot single-sided frequency spectrum
# ------------------------------------------------------------

def plot_spectrum(signal, title, ax):
    spectrum = np.fft.rfft(signal)
    frequency_axis = np.fft.rfftfreq(len(signal), d=1 / fs)

    # Normalize FFT magnitude
    magnitude = np.abs(spectrum) / len(signal)

    # Convert to single-sided amplitude spectrum
    if len(signal) % 2 == 0:
        magnitude[1:-1] *= 2
    else:
        magnitude[1:] *= 2

    ax.plot(frequency_axis / 1000, magnitude)
    ax.set_title(title)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True)
    ax.set_xlim(0, fs / 2000)


# ------------------------------------------------------------
# TASK 2: Add all four signals
# Plot the combined signal in time and frequency domains
# ------------------------------------------------------------

combined_signal = sum(signals.values())

fig, axes = plt.subplots(2, 1, figsize=(11, 8))

axes[0].plot(t * 1000, combined_signal)
axes[0].set_title("Combined Signal - Time Domain")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

plot_spectrum(
    combined_signal,
    "Combined Signal - Frequency Domain",
    axes[1]
)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# TASK 3: Butterworth digital low-pass filter
# Cutoff frequency = 2 kHz
# ------------------------------------------------------------

butterworth_cutoff = 2000

# Design a digital Butterworth low-pass filter
b_butter, a_butter = butter(
    filter_order,
    butterworth_cutoff,
    btype="low",
    fs=fs
)

# Filter the combined signal
butterworth_output = lfilter(
    b_butter,
    a_butter,
    combined_signal
)

print("\n========== BUTTERWORTH FILTER ==========")
print("Filter order:", filter_order)
print("Cutoff frequency:", butterworth_cutoff, "Hz")
print("Numerator coefficients:", b_butter)
print("Denominator coefficients:", a_butter)

fig, axes = plt.subplots(2, 1, figsize=(11, 8))

axes[0].plot(t * 1000, butterworth_output)
axes[0].set_title("Butterworth Output - Time Domain (2 kHz)")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

plot_spectrum(
    butterworth_output,
    "Butterworth Output - Frequency Domain",
    axes[1]
)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# TASK 4: Chebyshev Type-I digital low-pass filter
# Cutoff frequency = 3 kHz
# ------------------------------------------------------------

chebyshev_cutoff = 3000

# Design a digital Chebyshev Type-I low-pass filter
b_cheby, a_cheby = cheby1(
    filter_order,
    ripple,
    chebyshev_cutoff,
    btype="low",
    fs=fs
)

# Filter the same combined signal
chebyshev_output = lfilter(
    b_cheby,
    a_cheby,
    combined_signal
)

print("\n========== CHEBYSHEV TYPE-I FILTER ==========")
print("Filter order:", filter_order)
print("Cutoff frequency:", chebyshev_cutoff, "Hz")
print("Passband ripple:", ripple, "dB")
print("Numerator coefficients:", b_cheby)
print("Denominator coefficients:", a_cheby)

fig, axes = plt.subplots(2, 1, figsize=(11, 8))

axes[0].plot(t * 1000, chebyshev_output)
axes[0].set_title("Chebyshev Output - Time Domain (3 kHz)")
axes[0].set_xlabel("Time (ms)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

plot_spectrum(
    chebyshev_output,
    "Chebyshev Output - Frequency Domain",
    axes[1]
)

plt.tight_layout()
plt.show()



#  Compare the two digital filter responses

w_b, h_b = freqz(b_butter, a_butter, worN=2048, fs=fs)
w_c, h_c = freqz(b_cheby, a_cheby, worN=2048, fs=fs)

plt.figure(figsize=(11, 6))

plt.plot(w_b / 1000, 20 * np.log10(np.maximum(np.abs(h_b), 1e-10)),
         label="Butterworth (2 kHz cutoff)")

plt.plot(w_c / 1000, 20 * np.log10(np.maximum(np.abs(h_c), 1e-10)),
         label="Chebyshev Type-I (3 kHz cutoff)")

plt.title("Comparison of Digital Low-Pass Filter Responses")
plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.legend()
plt.xlim(0, fs / 2000)
plt.ylim(-100, 5)

plt.tight_layout()
plt.show()

print("\nAll additional tasks completed successfully!")
