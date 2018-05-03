import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# cells image reading and show
img = Image.open('cells.bmp')
img = img.convert('L')
img = np.array(img)
plt.figure(1)
plt.imshow(img, 'gray')
plt.title('Grey-scale Map')

# show histogaram
bins = np.arange(256)
hist, _ = np.histogram(img, np.hstack((bins, np.array([256]))))
plt.figure(2)
plt.bar(bins, hist)
plt.title('Histogram')

# solve otsu threshold
N = img.size
hist_norm = hist / N
max_delta2 = 0
for T in range(255):
    mu0 = 0
    mu1 = 0
    omega0 = np.sum(hist_norm[0:T+1])
    omega1 = 1-omega0
    for i in range(T+1):
        mu0 = mu0 + i * hist_norm[i]
    if omega0 != 0:
        mu0 = mu0 / omega0
    for i in range(T+1, 256):
        mu1 = mu1 + i * hist_norm[i]
    if omega1 != 0:
        mu1 = mu1 / omega1
    delta2 = omega0 * omega1 * (mu0-mu1)**2
    if max_delta2 < delta2:
        max_delta2 = delta2
        threshold = T
print('the otsu threshold is', threshold)

# image segmentation
img[img > threshold] = 255
img[img != 255] = 0
plt.figure(3)
plt.imshow(img,'gray')
plt.title('Segmentation Picture')

# show all
plt.show()
