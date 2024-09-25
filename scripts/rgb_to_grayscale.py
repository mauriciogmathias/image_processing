import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("../images/lenna.png")

#change image from BGR space to RGB space
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#split the image into R, G, B channels
R, G, B = cv2.split(img)

#create blank channels (zeros) to isolate each color channel
zeros = np.zeros_like(R)

#merge channels to create red, green, and blue images
R_img = cv2.merge([R, zeros, zeros])
G_img = cv2.merge([zeros, G, zeros])
B_img = cv2.merge([zeros, zeros, B])
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#plot the original image and each color channel in its respective color
plt.figure(figsize=(15, 5))

plt.subplot(1, 5, 1)
plt.imshow(img)
plt.title('original')

plt.subplot(1, 5, 2)
plt.imshow(R_img)
plt.title('red channel')

plt.subplot(1, 5, 3)
plt.imshow(G_img)
plt.title('green channel')

plt.subplot(1, 5, 4)
plt.imshow(B_img)
plt.title('blue channel')

plt.subplot(1, 5, 5)
plt.imshow(gray_img, cmap='gray')
plt.title('gray scale')

plt.tight_layout()
plt.show()
