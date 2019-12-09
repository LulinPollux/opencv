import numpy as np
import cv2
from matplotlib import pyplot as plt


def plt_arch(plot_number, title, src):
    plt.subplot(plot_number), plt.imshow(src, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])


def prewitt():
    src = cv2.imread('../res/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    mx = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]], dtype=np.float32)
    my = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]], dtype=np.float32)

    dx = cv2.filter2D(src, -1, mx)
    dy = cv2.filter2D(src, -1, my)
    dxdy = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)

    plt_arch(221, 'Input Image', src)
    plt_arch(222, 'dx', dx)
    plt_arch(223, 'dy', dy)
    plt_arch(224, 'dx + dy', dxdy)
    plt.show()


def sobel():
    src = cv2.imread('../res/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    mx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    my = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]], dtype=np.float32)

    dx = cv2.filter2D(src, -1, mx)
    dy = cv2.filter2D(src, -1, my)
    dxdy = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)

    plt_arch(221, 'Input Image', src)
    plt_arch(222, 'dx', dx)
    plt_arch(223, 'dy', dy)
    plt_arch(224, 'dx + dy', dxdy)
    plt.show()


def sobel_cv():
    src = cv2.imread('../res/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    dx = cv2.Sobel(src, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(src, cv2.CV_32F, 0, 1)

    fmag = cv2.magnitude(dx, dy)
    mag = np.clip(fmag, 0, 255)

    plt_arch(221, 'Input Image', src)
    plt_arch(222, 'dx', dx)
    plt_arch(223, 'dy', dy)
    plt_arch(224, 'mag', mag)
    plt.show()


if __name__ == '__main__':
    prewitt()
    sobel()
    sobel_cv()
