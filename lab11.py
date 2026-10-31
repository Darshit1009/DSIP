# AIM: Simulate dilation, erosion, opening and closing operation on images.
import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread('images.jpg', 0)

if image is None:
    raise FileNotFoundError("Image not found. Check the image path.")

# Define the kernel (structuring element)
kernel = np.ones((5, 5), np.uint8)

# Perform Erosion
erosion = cv2.erode(image, kernel, iterations=1)

# Perform Dilation
dilation = cv2.dilate(image, kernel, iterations=1)

# Perform Opening (Erosion followed by Dilation)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Perform Closing (Dilation followed by Erosion)
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Display the original image and results
cv2.imshow('Original Image', image)
cv2.imshow('Erosion', erosion)
cv2.imshow('Dilation', dilation)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()