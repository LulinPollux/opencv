import cv2
import numpy as np
from matplotlib import pyplot as plt

# 입력 영상을 랜덤값으로 만든다.
src = np.random.randn(100, 100)

# 커널을 만든다.
kernel = (1 / 9) * np.ones((3, 3))

# 2차원의 컨벌루션을 수행한다.
dst = cv2.filter2D(src, -1, kernel)

# 결과물을 출력한다.
plt.subplot(121), plt.imshow(src, cmap='gray')
plt.title('src'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(dst, cmap='gray')
plt.title('dst'), plt.xticks([]), plt.yticks([])
plt.show()
