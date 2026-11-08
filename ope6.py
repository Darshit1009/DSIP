
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Get the folder where this Python file is located
folder = os.path.dirname(os.path.abspath(__file__))

# Image names
images = ["p1.png", "p2.jpg", "p3.png"]

# Structuring elements
kernel_open = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)
)

kernel_close = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5, 5)
)

# Process each image
for name in images:

    path = os.path.join(folder, name)

    # Read image in grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Image not found:", name)
        continue

    # Invert p1 because its main object is black
    if name == "p1.jpg":
        img = cv2.bitwise_not(img)

    # Step 1: Opening - Remove small bright noise
    opening = cv2.morphologyEx(
        img,
        cv2.MORPH_OPEN,
        kernel_open
    )

    # Step 2: Closing - Fill narrow dark cracks
    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        kernel_close
    )

    # Restore original polarity for p1
    if name == "p1.jpg":
        closing = cv2.bitwise_not(closing)

    # Display results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(
        cv2.bitwise_not(img) if name == "p1.jpg" else img,
        cmap="gray"
    )
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(opening, cmap="gray")
    plt.title("After Opening")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(closing, cmap="gray")
    plt.title("After Opening + Closing")
    plt.axis("off")

    plt.suptitle("Morphological Processing - " + name)
    plt.tight_layout()

    # Save final processed image
    output_name = "cleaned_" + name
    cv2.imwrite(os.path.join(folder, output_name), closing)

    plt.show()

    print("Processed successfully:", name)
    print("Saved as:", output_name)
