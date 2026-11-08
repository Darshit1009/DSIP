
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.ndimage import generic_filter

# =====================================================
# OPE 5: NON-LINEAR FILTERING
# =====================================================

# 1. LOAD THREE IMAGES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_files = [
    "Picture1.png",
    "Picture2.png",
    "Picture3.png"
]

images = {}

for filename in image_files:

    image_path = os.path.join(BASE_DIR, filename)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    images[filename] = img

    print(f"Loaded: {filename}, Shape: {img.shape}")


# 2. DISPLAY ORIGINAL IMAGES

plt.figure(figsize=(15, 5))

for i, (name, img) in enumerate(images.items()):

    plt.subplot(1, 3, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(name)
    plt.axis("off")

plt.suptitle("Original Noisy Images")
plt.tight_layout()
plt.show()


# 3. NOISE ANALYSIS

def analyze_noise(img):

    total = img.size

    salt = np.sum(img >= 250)
    pepper = np.sum(img <= 5)

    local_mean = cv2.blur(img, (3, 3))

    residual = (
        img.astype(float) -
        local_mean.astype(float)
    )

    return {
        "Mean": round(np.mean(img), 2),
        "Standard Deviation": round(np.std(img), 2),
        "Salt Pixels (%)": round(salt / total * 100, 2),
        "Pepper Pixels (%)": round(pepper / total * 100, 2),
        "Noise Estimate": round(np.std(residual), 2)
    }


noise_data = []

for name, img in images.items():

    result = analyze_noise(img)
    result["Image"] = name

    noise_data.append(result)


noise_df = pd.DataFrame(noise_data)

print("\nNOISE ANALYSIS")
print(noise_df.to_string(index=False))


# 4. MEDIAN FILTERING

def median_filter(img, size):

    return cv2.medianBlur(img, size)


# 5. ALPHA-TRIMMED MEAN FILTERING

def alpha_trimmed_filter(img, size=3, d=2):

    def process(window):

        sorted_values = np.sort(window)

        trimmed = sorted_values[
            d // 2 : len(sorted_values) - d // 2
        ]

        return np.mean(trimmed)

    result = generic_filter(
        img.astype(np.float32),
        process,
        size=size,
        mode="reflect"
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# 6. MORPHOLOGICAL FILTERING

def morphological_filter(img, size=3):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (size, size)
    )

    opening = cv2.morphologyEx(
        img,
        cv2.MORPH_OPEN,
        kernel
    )

    closing = cv2.morphologyEx(
        img,
        cv2.MORPH_CLOSE,
        kernel
    )

    return opening, closing


# 7. APPLY ALL FILTERS

all_results = {}

for name, img in images.items():

    results = {
        "Original": img
    }

    # Median filters
    for size in [3, 5, 7]:

        results[f"Median {size}x{size}"] = median_filter(
            img, size
        )

    # Alpha-trimmed mean filters
    for size in [3, 5]:

        results[f"Alpha Trimmed {size}x{size}"] = (
            alpha_trimmed_filter(img, size, 2)
        )

    # Morphological filters
    opening, closing = morphological_filter(img)

    results["Morphological Opening"] = opening
    results["Morphological Closing"] = closing

    all_results[name] = results


# 8. DISPLAY FILTERED IMAGES

for name, results in all_results.items():

    plt.figure(figsize=(16, 10))

    for i, (method, filtered) in enumerate(results.items()):

        plt.subplot(3, 3, i + 1)

        plt.imshow(filtered, cmap="gray")
        plt.title(method)
        plt.axis("off")

    plt.suptitle(
        f"Non-Linear Filtering Results - {name}",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


# 9. PERFORMANCE ANALYSIS
# OpenCV Sobel - No Skimage Required

def calculate_metrics(original, filtered):

    original = original.astype(np.float32)
    filtered = filtered.astype(np.float32)

    # Noise estimation
    original_local = cv2.blur(original, (3, 3))
    filtered_local = cv2.blur(filtered, (3, 3))

    original_residual = original - original_local
    filtered_residual = filtered - filtered_local

    noise_before = np.std(original_residual)
    noise_after = np.std(filtered_residual)

    noise_reduction = (
        (noise_before - noise_after) /
        (noise_before + 1e-8)
    ) * 100

    # Edge detection using OpenCV Sobel

    original_x = cv2.Sobel(
        original, cv2.CV_32F, 1, 0, ksize=3
    )

    original_y = cv2.Sobel(
        original, cv2.CV_32F, 0, 1, ksize=3
    )

    filtered_x = cv2.Sobel(
        filtered, cv2.CV_32F, 1, 0, ksize=3
    )

    filtered_y = cv2.Sobel(
        filtered, cv2.CV_32F, 0, 1, ksize=3
    )

    original_edges = cv2.magnitude(
        original_x, original_y
    )

    filtered_edges = cv2.magnitude(
        filtered_x, filtered_y
    )

    edge_before = np.mean(original_edges)
    edge_after = np.mean(filtered_edges)

    edge_preservation = (
        edge_after / (edge_before + 1e-8)
    ) * 100

    return (
        round(noise_reduction, 2),
        round(edge_preservation, 2)
    )


# 10. FILTER PERFORMANCE COMPARISON

comparison = []

for name, results in all_results.items():

    original = results["Original"]

    for method, filtered in results.items():

        if method == "Original":
            continue

        noise, edge = calculate_metrics(
            original, filtered
        )

        comparison.append({
            "Image": name,
            "Filter": method,
            "Noise Reduction (%)": noise,
            "Edge Preservation (%)": edge
        })


comparison_df = pd.DataFrame(comparison)

print("\nFILTER PERFORMANCE COMPARISON")
print(comparison_df.to_string(index=False))


# 11. NOISE REDUCTION GRAPHS

for name in images.keys():

    data = comparison_df[
        comparison_df["Image"] == name
    ]

    plt.figure(figsize=(12, 5))

    plt.bar(
        data["Filter"],
        data["Noise Reduction (%)"]
    )

    plt.title(f"Noise Reduction - {name}")
    plt.xlabel("Filtering Method")
    plt.ylabel("Noise Reduction (%)")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


# 12. EDGE PRESERVATION GRAPHS

for name in images.keys():

    data = comparison_df[
        comparison_df["Image"] == name
    ]

    plt.figure(figsize=(12, 5))

    plt.bar(
        data["Filter"],
        data["Edge Preservation (%)"]
    )

    plt.title(f"Edge Preservation - {name}")
    plt.xlabel("Filtering Method")
    plt.ylabel("Edge Preservation (%)")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


# 13. BEST FILTER ANALYSIS

best_results = []

for name in images.keys():

    data = comparison_df[
        comparison_df["Image"] == name
    ].copy()

    data["Combined Score"] = (
        data["Noise Reduction (%)"] * 0.5 +
        data["Edge Preservation (%)"] * 0.5
    )

    best = data.loc[
        data["Combined Score"].idxmax()
    ]

    best_results.append({
        "Image": name,
        "Best Filter": best["Filter"],
        "Noise Reduction (%)": best["Noise Reduction (%)"],
        "Edge Preservation (%)": best["Edge Preservation (%)"],
        "Combined Score": round(best["Combined Score"], 2)
    })


best_df = pd.DataFrame(best_results)

print("\nBEST FILTER FOR EACH IMAGE")
print(best_df.to_string(index=False))


# 14. FINAL SUMMARY

print("\n======================================")
print("          OPE 5 COMPLETED")
print("======================================")

print("Total Images Processed:", len(images))
print("Filtering Methods Applied: 7")

print("\nNOISE ANALYSIS:")
print(noise_df.to_string(index=False))

print("\nFILTER COMPARISON:")
print(comparison_df.to_string(index=False))

print("\nFINAL BEST FILTER RESULTS:")
print(best_df.to_string(index=False))

print("\nAll outputs displayed successfully!")
