import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# -----------------------------
# Load Audio
# -----------------------------
audio_path = "C:/Users/hp/Downloads/sare-jahan-se-acha-beautifully-remastered-30952.mp3"

samples, sample_rate = librosa.load(audio_path, sr=None, mono=True)

# Normalize
samples = samples / np.max(np.abs(samples))

# -----------------------------
# Different Impulse Responses
# -----------------------------
impulse_responses = {
    "IR1": np.array([1, 0, 1, 0, 1], dtype=np.float32),
    "IR2": np.array([0.2, 0.5, 1, 0.5, 0.2], dtype=np.float32),
    "IR3": np.array([1, -0.5, 0.25], dtype=np.float32)
}

plt.figure(figsize=(12,8))

# -----------------------------
# Process Each IR
# -----------------------------
for index, (name, ir) in enumerate(impulse_responses.items()):

    # Convolution
    convoluted = fftconvolve(samples, ir, mode='same')

    convoluted /= np.max(np.abs(convoluted))

    # Save Convoluted Audio
    sf.write(f"{name}_convoluted.wav", convoluted, sample_rate)

    # -------------------------
    # Inverse Filtering
    # -------------------------

    ir_fft = np.fft.fft(ir, len(convoluted))

    conv_fft = np.fft.fft(convoluted)

    epsilon = 1e-6

    inverse_fft = conv_fft / (ir_fft + epsilon)

    restored = np.real(np.fft.ifft(inverse_fft))

    restored /= np.max(np.abs(restored))

    sf.write(f"{name}_restored.wav", restored, sample_rate)

    # Plot
    plt.subplot(3,1,index+1)
    plt.plot(restored[:5000])
    plt.title(f"{name} Restored Signal")

plt.tight_layout()
plt.show()

print("Processing Complete.")