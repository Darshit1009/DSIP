import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. GET THE CURRENT PYTHON FILE FOLDER
# ============================================================

folder = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 2. LOAD SOURCE IMAGE
# ============================================================

source_path = os.path.join(folder, "imageforlab8.jpg")

image = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(
        f"Source image not found:\n{source_path}"
    )


# ============================================================
# 3. CALCULATE ORIGINAL HISTOGRAM
# ============================================================

histogram = cv2.calcHist(
    [image],
    [0],
    None,
    [256],
    [0, 256]
)


# ============================================================
# 4. DISPLAY ORIGINAL HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 6))
plt.title("Original Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(histogram)
plt.xlim([0, 256])
plt.grid(True)
plt.show()


# ============================================================
# 5. HISTOGRAM EQUALIZATION
# ============================================================

equalized_image = cv2.equalizeHist(image)


# ============================================================
# 6. DISPLAY ORIGINAL AND EQUALIZED IMAGES
# ============================================================

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Equalized Image")
plt.imshow(equalized_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 7. CALCULATE EQUALIZED HISTOGRAM
# ============================================================

equalized_histogram = cv2.calcHist(
    [equalized_image],
    [0],
    None,
    [256],
    [0, 256]
)


# ============================================================
# 8. DISPLAY EQUALIZED HISTOGRAM
# ============================================================

plt.figure(figsize=(8, 6))
plt.title("Equalized Image Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.plot(equalized_histogram)
plt.xlim([0, 256])
plt.grid(True)
plt.show()


# ============================================================
# 9. LOAD REFERENCE IMAGE
# ============================================================

reference_path = os.path.join(folder, "imagebright.png")

reference_image = cv2.imread(
    reference_path,
    cv2.IMREAD_GRAYSCALE
)

if reference_image is None:
    raise FileNotFoundError(
        f"Reference image not found:\n{reference_path}"
    )


# ============================================================
# 10. CALCULATE SOURCE AND REFERENCE HISTOGRAMS
# ============================================================

source_hist = cv2.calcHist(
    [image],
    [0],
    None,
    [256],
    [0, 256]
)

reference_hist = cv2.calcHist(
    [reference_image],
    [0],
    None,
    [256],
    [0, 256]
)


# ============================================================
# 11. NORMALIZE HISTOGRAMS
# ============================================================

source_hist = source_hist / source_hist.sum()
reference_hist = reference_hist / reference_hist.sum()


# ============================================================
# 12. CALCULATE CUMULATIVE DISTRIBUTION FUNCTIONS (CDF)
# ============================================================

source_cdf = source_hist.cumsum()
reference_cdf = reference_hist.cumsum()


# ============================================================
# 13. PERFORM HISTOGRAM MATCHING
# ============================================================

mapping = np.interp(
    source_cdf,
    reference_cdf,
    np.arange(256)
)

matched_image = mapping[image].astype(np.uint8)


# ============================================================
# 14. DISPLAY SOURCE, REFERENCE AND MATCHED IMAGES
# ============================================================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Source Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Reference Image")
plt.imshow(reference_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Matched Image")
plt.imshow(matched_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 15. CALCULATE MATCHED IMAGE HISTOGRAM
# ============================================================

matched_hist = cv2.calcHist(
    [matched_image],
    [0],
    None,
    [256],
    [0, 256]
)

matched_hist = matched_hist / matched_hist.sum()


# ============================================================
# 16. COMPARE HISTOGRAMS
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    source_hist,
    label="Source Histogram"
)

plt.plot(
    reference_hist,
    label="Reference Histogram"
)

plt.plot(
    matched_hist,
    label="Matched Histogram"
)

plt.title("Histogram Matching")
plt.xlabel("Pixel Value")
plt.ylabel("Normalized Frequency")
plt.xlim([0, 256])
plt.grid(True)
plt.legend()

plt.show()