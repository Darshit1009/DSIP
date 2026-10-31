import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read image in grayscale
img = cv2.imread('bird.jpg', 0)

if img is None:
    raise FileNotFoundError("Image not found. Check the image path.")

# 2. Apply Fourier Transform
F = np.fft.fftshift(np.fft.fft2(img))
M, N = img.shape

# 3. Create frequency distance matrix
u, v = np.meshgrid(
    np.arange(N) - N // 2,
    np.arange(M) - M // 2
)
D = np.sqrt(u**2 + v**2)

# 4. Create Low-Pass and High-Pass Filters
LPF = (D < 30).astype(float)
HPF = (D >= 30).astype(float)

# 5. Apply filters and inverse Fourier Transform
smooth = np.abs(
    np.fft.ifft2(np.fft.ifftshift(F * LPF))
)

high_pass = np.abs(
    np.fft.ifft2(np.fft.ifftshift(F * HPF))
)

# 6. Sharpen the original image
sharp = np.clip(img.astype(float) + high_pass, 0, 255)

# 7. Display results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(smooth, cmap='gray')
plt.title('LPF - Smoothening')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sharp, cmap='gray')
plt.title('HPF - Sharpening')
plt.axis('off')

plt.tight_layout()
plt.show()