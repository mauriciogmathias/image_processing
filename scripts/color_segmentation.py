import cv2
import numpy as np
import matplotlib.pyplot as plt

#read the image
img = cv2.imread("../images/grass.png")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#convert the image from rgb to hsv color space
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

#define the hsv range for thresholding (green in this case)
lower_green = np.array([40, 40, 40])  #lower bound for green
upper_green = np.array([80, 255, 255])  #upper bound for green

#apply the threshold to create a mask for green color
mask = cv2.inRange(img_hsv, lower_green, upper_green)

#apply the mask on the original rgb image (optional, just to show the result)
result = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

#display the original image, hsv image, mask, and the threshold result
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(img_rgb)
plt.title("original image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(img_hsv)
plt.title("hsv image")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(mask, cmap='gray')
plt.title("mask")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(result)
plt.title("thresholded image (green)")
plt.axis("off")

plt.show()

#pure green
rgb_green = np.uint8([[[0, 255, 0]]])
hsv_green = cv2.cvtColor(rgb_green, cv2.COLOR_RGB2HSV)[0, 0, :]
print(f"hsv value of pure green: {hsv_green}")