import numpy as np
import matplotlib.pyplot as plt


def linear_convolution(signal1, signal2):
    # Compute the linear convolution
    linear_conv = np.convolve(signal1, signal2, mode='full')
    return linear_conv


def circular_convolution(signal1, signal2):
    # Compute the circular convolution
    if len(signal1) > len(signal2):
        fft_length = len(signal1)
    else:
        fft_length = len(signal2)

    fft_signal1 = np.fft.fft(signal1, fft_length)
    fft_signal2 = np.fft.fft(signal2, fft_length)

    circular_conv = np.fft.ifft(fft_signal1 * fft_signal2)
    return circular_conv


# Marwadi University
# Faculty of Engineering & Technology
# Department of Information and Communication Technology

# Subject: Digital Signal and Image Processing

# (01CT1513)

# AIM: Simulate linear convolution and circular convolution
# on discrete time signals.

# Experiment No: 2
# Date: 26-07-2026      
# Enrollment No: 92400133089

# Define the discrete-time signals
signal1 = np.array([1, 2, 3, 4, 5])
signal2 = np.array([2, 4, 6, 8, 10])

# Compute the linear convolution
linear_conv = linear_convolution(signal1, signal2)

# Compute the circular convolution
circular_conv = circular_convolution(signal1, signal2)

# Plot the results
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.stem(linear_conv)
plt.title('Linear Convolution')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.stem(np.real(circular_conv))
plt.title('Circular Convolution')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Optional: Save the results
# np.savetxt('linear_convolution.txt', linear_conv, delimiter=',')
# np.savetxt('circular_convolution.txt', np.real(circular_conv), delimiter=',')