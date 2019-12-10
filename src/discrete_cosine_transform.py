import cv2
import numpy as np
from matplotlib import pyplot as plt

# 원본 이미지를 입력한다.
img = cv2.imread('../res/lenna.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


""" 이산 코사인 변환(DCT) """
dct = cv2.dct(np.float32(img))  # 이산 코사인 변환을 수행한다.
spectrum = 20 * np.log(1 + cv2.magnitude(dct[:, :], dct[:, :]))  # 스펙트럼 영상을 구한다.

""" 역 이산 코사인 변환(IDCT) """
idct = cv2.idct(dct)  # 역 이산 코사인 변환을 수행한다.

""" 저주파 통과 필터링(LPF) """
rows, cols = img.shape
size = 256
# 왼쪽상단 = 1, 나머지 = 0
lpf_mask = np.zeros((rows, cols), np.uint8)
lpf_mask[0:size, 0:size] = 1
lpf = dct * lpf_mask  # 마스크를 적용한다. (요소별 곱셈: n x 1 = n, n x 0 = 0)
lpf_spectrum = 20 * np.log(1 + cv2.magnitude(lpf[:, :], lpf[:, :]))

""" 고주파 통과 필터링(HPF) """
rows, cols = img.shape
size = 40
# 왼쪽상단 = 0, 나머지 = 1
hpf_mask = np.ones((rows, cols), np.uint8)
hpf_mask[0:size, 0:size] = 0
hpf = dct * hpf_mask  # 마스크를 적용한다. (요소별 곱셈: n x 1 = n, n x 0 = 0)
hpf_spectrum = 20 * np.log(1 + cv2.magnitude(hpf[:, :], hpf[:, :]))

""" LPF, HPF가 적용된 역 이산 코사인 변환(LPF, HPF with IFFT)"""
lpf_idct = cv2.idct(lpf)
hpf_idct = cv2.idct(hpf)


# 결과물을 출력한다.
plt.subplot(331), plt.imshow(img, cmap='gray')  # 원본 영상
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(332), plt.imshow(spectrum, cmap='gray')  # DCT 영상
plt.title('DCT'), plt.xticks([]), plt.yticks([])

plt.subplot(333), plt.imshow(idct, cmap='gray')  # IDCT 영상
plt.title('IDCT'), plt.xticks([]), plt.yticks([])

plt.subplot(334), plt.imshow(lpf_idct, cmap='gray')  # LPF가 적용된 IDCT 영상
plt.title('IDCT(LPF apply)'), plt.xticks([]), plt.yticks([])

plt.subplot(335), plt.imshow(lpf_spectrum, cmap='gray')  # LPF가 적용된 DCT 영상
plt.title('DCT(LPF apply)'), plt.xticks([]), plt.yticks([])

plt.subplot(336), plt.imshow(lpf_mask, cmap='gray')  # LPF 마스크
plt.title('LPF Mask'), plt.xticks([]), plt.yticks([])

plt.subplot(337), plt.imshow(hpf_idct, cmap='gray')  # HPF가 적용된 IDCT 영상
plt.title('IDCT(HPF apply)'), plt.xticks([]), plt.yticks([])

plt.subplot(338), plt.imshow(hpf_spectrum, cmap='gray')  # HPF가 적용된 DCT 영상
plt.title('DCT(HPF apply)'), plt.xticks([]), plt.yticks([])

plt.subplot(339), plt.imshow(hpf_mask, cmap='gray')  # HPF 마스크
plt.title('HPF Mask'), plt.xticks([]), plt.yticks([])
plt.show()
