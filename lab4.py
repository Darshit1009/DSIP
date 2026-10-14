
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, bilinear, freqz, cheby1, lfilter

def design_butterworth_filter(filter_order, cutoff_frequency, sampling_frequency):
        analog_b, analog_a = butter(
            filter_order,
            cutoff_frequency,
            analog=True,
            btype='low'
        )

        digital_b, digital_a = bilinear(
            analog_b,
            analog_a,
            fs=sampling_frequency
        )

        return digital_b, digital_a


def design_chebyshev_filter(filter_order, cutoff_frequency, sampling_frequency, ripple):
        analog_b, analog_a = cheby1(
            filter_order,
            ripple,
            cutoff_frequency,
            analog=True,
            btype='low'
        )

        digital_b, digital_a = bilinear(
            analog_b,
            analog_a,
            fs=sampling_frequency
        )

        return digital_b, digital_a


def plot_filter_response(digital_b, digital_a, sampling_frequency):
        frequency, magnitude_response = freqz(
            digital_b,
            digital_a,
            fs=sampling_frequency
        )

        plt.figure(figsize=(10, 6))
        plt.plot(frequency, np.abs(magnitude_response))
        plt.title('Filter Magnitude Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()

        impulse = np.zeros(100)
        impulse[0] = 1

        impulse_response = lfilter(digital_b, digital_a, impulse)

        plt.figure(figsize=(10, 6))
        plt.stem(range(len(impulse_response)), impulse_response)
        plt.title('Filter Impulse Response')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()


filter_order = 4
cutoff_frequency = 1000
sampling_frequency = 8000
ripple = 0.5

digital_b, digital_a = design_butterworth_filter(
    filter_order,
    cutoff_frequency,
    sampling_frequency
)
print("Butterworth Filter Coefficients:")
print("Numerator:", digital_b)
print("Denominator:", digital_a)
plot_filter_response(digital_b, digital_a, sampling_frequency)
digital_b, digital_a = design_chebyshev_filter(
    filter_order,
    cutoff_frequency,
    sampling_frequency,
    ripple
)
print("Chebyshev Filter Coefficients:")
print("Numerator:", digital_b)
print("Denominator:", digital_a)

plot_filter_response(digital_b, digital_a, sampling_frequency)
filter_path = 'filter_coefficients.txt'
np.savetxt(filter_path, np.vstack((digital_b, digital_a)), delimiter=',')
print(f"Filter coefficients saved at: {filter_path}")