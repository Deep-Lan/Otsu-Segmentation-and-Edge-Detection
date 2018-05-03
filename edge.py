import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def convolve(img, filter, padding='same'):
    '''
    parameters: img is the image input
                filter is the kernel
                padding is the way to pad,there are only 2 ways,the one is 'same',the other is 'valid'
    return: Img is the respond of the image input
    '''
    if padding == 'same':
        img = np.pad(img, ((1, 1), (1, 1)), 'constant')
    row, column = img.shape
    Img = np.zeros((row-2, column-2))
    for i in range(row-2):
        for j in range(column-2):
            Img[i, j] = np.sum(img[i:i+3, j:j+3] * filter)
    return Img

def main():
    # lena image reading and show
    img = Image.open('lena.bmp')
    img = np.array(img)
    plt.figure(1)
    plt.imshow(img, 'gray')
    plt.title('original')

    # filter
    filt_laplace = np.array([0, 1, 0, 1, -4, 1, 0, 1, 0]).reshape((3, 3))
    filt_sobel_h1 = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]).reshape((3, 3))
    filt_sobel_h3 = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape((3, 3))
    filt_kirsch_h1 = np.array([3, 3, 3, 3, 0, 3, -5, -5, -5]).reshape((3, 3))
    filt_kirsch_h2 = np.array([3, 3, 3, -5, 0, 3, -5, -5, 3]).reshape((3, 3))
    filt_kirsch_h3 = np.array([-5, 3, 3, -5, 0, 3, -5, 3, 3]).reshape((3, 3))
    filt_kirsch_h4 = np.array([-5, -5, 3, -5, 0, 3, 3, 3, 3]).reshape((3, 3))
    filt_kirsch_h5 = np.array([-5, -5, -5, 3, 0, 3, 3, 3, 3]).reshape((3, 3))
    filt_kirsch_h6 = np.array([3, -5, -5, 3, 0, -5, 3, 3, 3]).reshape((3, 3))
    filt_kirsch_h7 = np.array([3, 3, -5, 3, 0, -5, 3, 3, -5]).reshape((3, 3))
    filt_kirsch_h8 = np.array([3, 3, 3, 3, 0, -5, 3, -5, -5]).reshape((3, 3))

    # convolve
    img_laplace = convolve(img, filt_laplace, padding='same')
    img_sobel_h1 = convolve(img, filt_sobel_h1, padding='same')
    img_sobel_h3 = convolve(img, filt_sobel_h3, padding='same')
    img_kirsch_h1 = convolve(img, filt_kirsch_h1, padding='same')
    img_kirsch_h2 = convolve(img, filt_kirsch_h2, padding='same')
    img_kirsch_h3 = convolve(img, filt_kirsch_h3, padding='same')
    img_kirsch_h4 = convolve(img, filt_kirsch_h4, padding='same')
    img_kirsch_h5 = convolve(img, filt_kirsch_h5, padding='same')
    img_kirsch_h6 = convolve(img, filt_kirsch_h6, padding='same')
    img_kirsch_h7 = convolve(img, filt_kirsch_h7, padding='same')
    img_kirsch_h8 = convolve(img, filt_kirsch_h8, padding='same')

    # find responds
    img_laplace = np.abs(img_laplace)
    img_sobel = np.abs(img_sobel_h1) + np.abs(img_sobel_h3)
    img_kirsch = np.max(np.array([img_kirsch_h1,
                                  img_kirsch_h2,
                                  img_kirsch_h3,
                                  img_kirsch_h4,
                                  img_kirsch_h5,
                                  img_kirsch_h6,
                                  img_kirsch_h7,
                                  img_kirsch_h8]), axis=0)

    # threshold the responds
    thre_laplace = 30
    img_laplace[img_laplace > thre_laplace] = 255
    img_laplace[img_laplace <= thre_laplace] = 0
    thre_sobel = 150
    img_sobel[img_sobel > thre_sobel] = 255
    img_sobel[img_sobel <= thre_sobel] = 0
    thre_kirsch = 240
    img_kirsch[img_kirsch > thre_kirsch] = 255
    img_kirsch[img_kirsch <= thre_kirsch] = 0

    # show the pictures of responds
    plt.figure(2)
    plt.imshow(img_laplace, 'gray')
    plt.title('laplace')
    plt.figure(3)
    plt.imshow(img_sobel, 'gray')
    plt.title('sobel')
    plt.figure(4)
    plt.imshow(img_kirsch, 'gray')
    plt.title('kirsch')
    plt.show()

if __name__ == '__main__':
    main()