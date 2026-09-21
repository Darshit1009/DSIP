import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread('imageforlab9.jpg')

if image is None:
    print("Error: Image not found!")
    exit()

# OpenCV reads BGR, Matplotlib displays RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# =========================================================
# 1. GAUSSIAN SMOOTHING
# =========================================================

kernel_size = (5, 5)
sigma = 1.5

# Create 1D Gaussian kernel
gaussian_kernel_1d = cv2.getGaussianKernel(5, sigma)

# Convert 1D kernel into 2D kernel
gaussian_kernel = np.outer(
    gaussian_kernel_1d,
    gaussian_kernel_1d
)

print("\n========== GAUSSIAN KERNEL ==========")
print(gaussian_kernel)

# Apply Gaussian smoothing
gaussian_image = cv2.filter2D(
    image,
    -1,
    gaussian_kernel
)

# Convert BGR to RGB
gaussian_rgb = cv2.cvtColor(
    gaussian_image,
    cv2.COLOR_BGR2RGB
)


# =========================================================
# 2. AVERAGING LINEAR FILTER
# =========================================================

kernel_size = (5, 5)

# Create 5 x 5 averaging kernel
average_kernel = np.ones(
    kernel_size,
    dtype=np.float32
) / (kernel_size[0] * kernel_size[1])

print("\n========== AVERAGING KERNEL ==========")
print(average_kernel)

# Apply averaging filter
average_image = cv2.filter2D(
    image,
    -1,
    average_kernel
)

# Convert BGR to RGB
average_rgb = cv2.cvtColor(
    average_image,
    cv2.COLOR_BGR2RGB
)


# =========================================================
# 3. MEDIAN FILTER
# =========================================================

# Kernel size must be odd
median_kernel_size = 5

# Apply median filter
median_image = cv2.medianBlur(
    image,
    median_kernel_size
)

# Convert BGR to RGB
median_rgb = cv2.cvtColor(
    median_image,
    cv2.COLOR_BGR2RGB
)


# =========================================================
# 4. SPATIAL HIGH-PASS / LAPLACIAN SHARPENING
# =========================================================

# Gaussian smoothing before sharpening
blurred_image = cv2.GaussianBlur(
    image,
    (5, 5),
    0
)

# Laplacian sharpening kernel
laplacian_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

print("\n========== LAPLACIAN KERNEL ==========")
print(laplacian_kernel)

# Apply Laplacian sharpening
laplacian_image = cv2.filter2D(
    blurred_image,
    -1,
    laplacian_kernel
)

# Convert images to RGB
blurred_rgb = cv2.cvtColor(
    blurred_image,
    cv2.COLOR_BGR2RGB
)

laplacian_rgb = cv2.cvtColor(
    laplacian_image,
    cv2.COLOR_BGR2RGB
)


# =========================================================
# DISPLAY ALL RESULTS
# SCREEN-FRIENDLY: 2 COLUMNS × 3 ROWS
# =========================================================

plt.figure(
    figsize=(14, 18),
    dpi=100
)


# ---------------------------------------------------------
# 1. ORIGINAL IMAGE
# ---------------------------------------------------------

plt.subplot(3, 2, 1)

plt.imshow(image_rgb)

plt.title(
    "1. Original Image",
    fontsize=18,
    fontweight="bold",
    pad=15
)

plt.axis("off")


# ---------------------------------------------------------
# 2. GAUSSIAN SMOOTHING
# ---------------------------------------------------------

plt.subplot(3, 2, 2)

plt.imshow(gaussian_rgb)

plt.title(
    "2. Gaussian Smoothing",
    fontsize=18,
    fontweight="bold",
    pad=15
)

plt.axis("off")


# ---------------------------------------------------------
# 3. AVERAGING FILTER
# ---------------------------------------------------------

plt.subplot(3, 2, 3)

plt.imshow(average_rgb)

plt.title(
    "3. Averaging Filter",
    fontsize=18,
    fontweight="bold",
    pad=15
)

plt.axis("off")


# ---------------------------------------------------------
# 4. MEDIAN FILTER
# ---------------------------------------------------------

plt.subplot(3, 2, 4)

plt.imshow(median_rgb)

plt.title(
    "4. Median Filter",
    fontsize=18,
    fontweight="bold",
    pad=15
)

plt.axis("off")


# ---------------------------------------------------------
# 5. GAUSSIAN BLURRED IMAGE
# ---------------------------------------------------------

plt.subplot(3, 2, 5)

plt.imshow(blurred_rgb)

plt.title(
    "5. Gaussian Blurred",
    fontsize=18,
    fontweight="bold",
    pad=15
)

plt.axis("off")


# ---------------------------------------------------------
# 6. LAPLACIAN SHARPENING
# ---------------------------------------------------------

plt.subplot(3, 2, 6)

plt.imshow(laplacian_rgb)

plt.title(
    "6. Laplacian Sharpening",
    fontsize=18,
    fontweight="bold",
    pad=15
)

plt.axis("off")


# Adjust spacing
plt.tight_layout(
    pad=4.0
)

# Display
plt.show()