import numpy as np
import cv2
from matplotlib import pyplot as plt


def plt_arch(plot_number, title, src):
    plt.subplot(plot_number), plt.imshow(src, cmap='gray')
    plt.title(title), plt.xticks([]), plt.yticks([])


def roberts():
    src = cv2.imread('../res/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    m_row = np.array([[-1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]], dtype=np.float32)
    m_col = np.array([[0, 0, -1],
                      [0, 1, 0],
                      [0, 0, 0]], dtype=np.float32)

    d_row = cv2.filter2D(src, cv2.CV_32F, m_row)
    d_col = cv2.filter2D(src, cv2.CV_32F, m_col)
    compound = cv2.magnitude(d_row, d_col)
    gray_compound = np.clip(compound, 0, 255)

    plt_arch(231, 'Input Image', src)
    plt_arch(232, 'D_row', d_row)
    plt_arch(233, 'D_col', d_col)
    plt_arch(234, 'Compound', gray_compound)
    plt.show()


def prewitt():
    src = cv2.imread('../res/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    m_row = np.array([[-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]], dtype=np.float32)
    m_col = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0, -1]], dtype=np.float32)

    d_row = cv2.filter2D(src, cv2.CV_32F, m_row)
    d_col = cv2.filter2D(src, cv2.CV_32F, m_col)
    compound = cv2.magnitude(d_row, d_col)
    gray_compound = np.clip(compound, 0, 255)

    plt_arch(231, 'Input Image', src)
    plt_arch(232, 'D_row', d_row)
    plt_arch(233, 'D_col', d_col)
    plt_arch(234, 'Compound', gray_compound)
    plt.show()


def sobel():
    src = cv2.imread('../res/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    m_row = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]], dtype=np.float32)
    m_col = np.array([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]], dtype=np.float32)

    d_row = cv2.filter2D(src, cv2.CV_32F, m_row)
    d_col = cv2.filter2D(src, cv2.CV_32F, m_col)
    compound = cv2.magnitude(d_row, d_col)
    gray_compound = np.clip(compound, 0, 255)

    plt_arch(231, 'Input Image', src)
    plt_arch(232, 'D_row', d_row)
    plt_arch(233, 'D_col', d_col)
    plt_arch(234, 'Compound', gray_compound)
    plt.show()


def sobel_cv():
    src = cv2.imread('../res/lenna.bmp', cv2.IMREAD_GRAYSCALE)

    d_row = cv2.Sobel(src, cv2.CV_32F, 0, 1)
    d_col = cv2.Sobel(src, cv2.CV_32F, 1, 0)

    compound = cv2.magnitude(d_col, d_row)
    gray_compound = np.clip(compound, 0, 255)

    plt_arch(231, 'Input Image', src)
    plt_arch(232, 'D_row', d_row)
    plt_arch(233, 'D_col', d_col)
    plt_arch(234, 'Compound', gray_compound)
    plt.show()


if __name__ == '__main__':
    roberts()
    prewitt()
    sobel()
