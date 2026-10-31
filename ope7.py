import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Current project folder
input_folder = r"D:\SEM_5\DSIP"
output_folder = os.path.join(input_folder, "enhanced_images")
os.makedirs(output_folder, exist_ok=True)

image_names = [
    "Picture1.jpg",
    "Picture2.jpg",
    "Picture3.png",
    "Picture4.png"
]

for name in image_names:
    path = os.path.join(input_folder, name)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Image not found: {path}")
        continue

    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    if p98 > p2:
        stretched = np.clip(
            (img.astype(np.float32) - p2) * 255 / (p98 - p2),
            0, 255
        ).astype(np.uint8)
    else:
        stretched = img.copy()

    # Histogram equalization
    equalized = cv2.equalizeHist(img)

    # Gamma correction to brighten dark images
    gamma = 3.0
    table = np.array([
        ((i / 255.0) ** (1.0 / gamma)) * 255
        for i in range(256)
    ]).clip(0, 255).astype(np.uint8)

    gamma_img = cv2.LUT(img, table)

    # CLAHE enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    clahe_img = clahe.apply(img)

    # Display original and enhanced results
    images = [
        ("Original", img),
        ("Contrast Stretching", stretched),
        ("Histogram Equalization", equalized),
        ("Gamma Correction", gamma_img),
        ("CLAHE", clahe_img)
    ]

    plt.figure(figsize=(14, 7))

    for i, (title, result) in enumerate(images):
        plt.subplot(2, 3, i + 1)
        plt.imshow(result, cmap="gray", vmin=0, vmax=255)
        plt.title(title)
        plt.axis("off")

        if i > 0:
            output_name = (
                os.path.splitext(name)[0]
                + "_" + title.lower().replace(" ", "_")
                + ".png"
            )
            cv2.imwrite(
                os.path.join(output_folder, output_name),
                result
            )

    plt.suptitle(name)
    plt.tight_layout()
    plt.show()

    print(f"Successfully processed: {name}")

print("Processing completed!")