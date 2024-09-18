import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

img = cv2.imread("images/lenna.png")

#prepare a 11x11 averaging filter
kernel = np.ones((11, 11), np.float32)/121
dst = cv2.filter2D(img, -1, kernel)

#change image from BGR space to RGB space
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(dst), plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#perform laplacian filtering
laplacian = cv2.Laplacian(img, cv2.CV_64F)

#find vertical edge
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

#find horizontal edge
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

#display the results
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original (GS)'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()