import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
image = cv2.imread("C:/Users/hp/OneDrive/Pictures/Screenshots/Screenshot 2026-07-26 164051.png", 1)

# Convert BGR to RGB    
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display original image
plt.figure(figsize=(6, 6))
plt.imshow(image_rgb)
plt.axis("off")
plt.show()

# Print image shape
print(image.shape)

# Split BGR channels
b, g, r = cv2.split(image)

# Create 4 plots
fig, ax = plt.subplots(1, 4, figsize=(16, 4))

# Original
ax[0].imshow(image_rgb)
ax[0].set_title("Original")
ax[0].axis("on")

# Red
ax[1].imshow(r, cmap="Reds")
ax[1].set_title("Red")
ax[1].axis("on")

# Green
ax[2].imshow(g, cmap="Greens")
ax[2].set_title("Green")
ax[2].axis("on")

# Blue
ax[3].imshow(b, cmap="Blues")
ax[3].set_title("Blue")
ax[3].axis("on")

plt.tight_layout()
plt.show()

crop=image[10:100,20:350]
plt.figure(figsize=(10,2))
plt.subplot(121)
plt.imshow(crop)
plt.axis('off')
