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


""" 고주파 통과 필터링(HPF) """
rows, cols = img.shape
size = 23
mask = np.ones((rows, cols), np.uint8)
mask[0:size, 0:size] = 0
hpf = dct * mask  # 마스크를 적용한다. (요소별 곱셈: n x 1 = n, n x 0 = 0)
hpf_idct = cv2.idct(hpf)

""" 저주파 통과 필터링(LPF) """
rows, cols = img.shape
size = 256
mask = np.zeros((rows, cols), np.uint8)
mask[0:size, 0:size] = 1
lpf = dct * mask  # 마스크를 적용한다. (요소별 곱셈: n x 1 = n, n x 0 = 0)
lpf_idct = cv2.idct(lpf)

# 결과물을 출력한다.
plt.subplot(231), plt.imshow(img, cmap='gray')  # 원본 영상
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(232), plt.imshow(spectrum_img, cmap='gray')  # DCT 영상
plt.title('DCT'), plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(idct, cmap='gray')  # IDCT 영상
plt.title('IDCT'), plt.xticks([]), plt.yticks([])

plt.subplot(235), plt.imshow(lpf_idct, cmap='gray')  # LPF가 적용된 IFFT 영상
plt.title('IDCT(LPF apply)'), plt.xticks([]), plt.yticks([])

plt.subplot(236), plt.imshow(hpf_idct, cmap='gray')  # HPF가 적용된 IFFT 영상
plt.title('IDCT(HPF apply)'), plt.xticks([]), plt.yticks([])
plt.show()
