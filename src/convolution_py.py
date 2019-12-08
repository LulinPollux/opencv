import numpy as np
from matplotlib import pyplot as plt


def conv2D(src, kernel):
    s_rows, s_cols = src.shape      # 입력 영상의 형태
    k_rows, k_cols = kernel.shape   # 커널의 형태
    d = k_rows // 2             # 경계부분 처리, 커널 연산에 사용하는 변수
    dst = np.ones(src.shape)

    for s_row in range(d, s_rows - d):
        for s_col in range(d, s_cols - d):
            # convolution 처리
            temp = 0
            for k_row in range(k_rows):
                for k_col in range(k_cols):
                    temp += src[s_row + (k_row - d)][s_col + (k_col - d)] * kernel[k_row][k_col]
            dst[s_row][s_col] = temp
    return dst


# 입력 영상을 랜덤값으로 만든다.
src = np.random.rand(100, 100)

# 커널을 만든다.
kernel = (1 / 9) * np.ones((3, 3))

# 2차원의 컨벌루션을 수행한다.
dst = conv2D(src, kernel)

# 결과물을 출력한다.
plt.subplot(121), plt.imshow(src, cmap='gray')
plt.title('src'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(dst, cmap='gray')
plt.title('dst'), plt.xticks([]), plt.yticks([])
plt.show()
