import cv2
import numpy as np
from matplotlib import pyplot as plt

# 원본 이미지를 입력한다.
img = cv2.imread('../res/lenna.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

""" 이산 코사인 변환(DCT) """
dct = cv2.dct(np.float32(img))  # 이산 코사인 변환을 수행한다.
spectrum_img = 20 * np.log(cv2.magnitude(dct[:, :], dct[:, :]))  # 스펙트럼 영상을 구한다.

""" 역 이산 코사인 변환(IDCT) """
idct = cv2.idct(dct)

# 결과물을 출력한다.
plt.subplot(221), plt.imshow(img, cmap='gray')  # 원본 영상
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(spectrum_img, cmap='gray')  # DCT 영상
plt.title('DCT'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(idct, cmap='gray')  # IDCT 영상
plt.title('IDCT'), plt.xticks([]), plt.yticks([])
plt.show()
