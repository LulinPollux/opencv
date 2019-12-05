import cv2
import numpy as np
from matplotlib import pyplot as plt

# 원본 이미지를 입력한다.
img = cv2.imread('../res/lenna.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

""" 이산 푸리에 변환(DFT) """
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)  # 이산 푸리에 변환을 적용한다.
dft_shift = np.fft.fftshift(dft)  # 셔플링을 한다.
# 스펙트럼 영상을 구한다. D(u, v) = c x log( |F(u, v)| )
spectrum_img = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

""" 역 이산 푸리에 변환(IDFT) """
dft_ishift = np.fft.ifftshift(dft_shift)  # 셔플링 되었던 것을 역셔플링한다.
idft = cv2.idft(dft_ishift)  # 역 이산 푸리에 변환을 한다.
idft = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])  # 절대값 적용

""" 저주파 통과 필터링(LPF) """
rows, cols = img.shape
center_row, center_col = rows // 2, cols // 2  # 이미지의 중심 좌표
size = 20  # 통과시킬 저주파 영역의 크기 (n x n)
# 가운데 = 1, 외곽 = 0
mask = np.zeros((rows, cols, 2), np.uint8)
mask[center_row - size:center_row + size, center_col - size:center_col + size] = 1
lpf = dft_shift * mask  # 마스크를 적용한다. (요소별 곱셈: n x 1 = n, n x 0 = 0)

""" LPF가 적용된 역 이산 푸리에 변환(LPF with IDFT) """
dft_ishift2 = np.fft.ifftshift(lpf)  # 셔플링 되었던 것을 역셔플링한다.
idft2 = cv2.idft(dft_ishift2)  # 역 이산 푸리에 변환을 한다.
idft2 = cv2.magnitude(idft2[:, :, 0], idft2[:, :, 1])  # 절대값 적용

# 결과물을 출력한다.
plt.subplot(221), plt.imshow(img, cmap='gray')  # 원본 영상
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(spectrum_img, cmap='gray')  # DFT 영상
plt.title('DFT'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(idft, cmap='gray')  # IDFT 영상
plt.title('IDFT'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(idft2, cmap='gray')  # LPF가 적용된 IDFT 영상
plt.title('IDFT(LPF apply)'), plt.xticks([]), plt.yticks([])
plt.show()
