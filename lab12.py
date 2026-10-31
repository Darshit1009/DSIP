import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread('images.jpg', 0)

if image is None:
    raise FileNotFoundError("Image not found. Check the image path.")

# Convert image into binary format
# Black objects become foreground (1)
_, binary = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY_INV)

# Define the structuring element
#  1 = Foreground
# -1 = Background
#  0 = Don't care
pattern = np.array([
    [-1,  1, -1],
    [ 1,  1,  1],
    [-1,  1, -1]
], dtype=np.int8)

# Apply Hit-or-Miss Transformation
result = cv2.morphologyEx(binary, cv2.MORPH_HITMISS, pattern)

# Convert result into 0-255 format for display
result_display = (result * 255).astype(np.uint8)

# Display images
cv2.imshow('Original Binary Image', binary * 255)
cv2.imshow('Hit or Miss Transformation', result_display)

# Wait and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()