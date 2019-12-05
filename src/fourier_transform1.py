import cv2
import numpy as np
from matplotlib import pyplot as plt

# 원본 이미지를 입력한다.
img = cv2.imread('../res/lenna.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

""" 이산 푸리에 변환(DFT)
    적용을 하면 화면 좌측상단(0, 0)이 중심이고, 그곳에 저주파가 모여 있다.
    분석을 쉽게 하기 위해 셔플링을 하고 Log Scaling을 하여 매우 넓은 범위의 값을 축소시킨다. """
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)  # 이산 푸리에 변환을 적용한다.
dft_shift = np.fft.fftshift(dft)  # 셔플링을 한다.
# 스펙트럼 영상을 구한다. D(u, v) = c x log( |F(u, v)| )
spectrum = 20 * np.log(1 + cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

""" 역 이산 푸리에 변환(IDFT) """
dft_ishift = np.fft.ifftshift(dft_shift)  # 셔플링 되었던 것을 역셔플링한다.
idft = cv2.idft(dft_ishift)  # 역 이산 푸리에 변환을 한다.
idft = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])  # 절대값 적용

""" 저주파 통과 필터링(LPF) """
rows, cols = img.shape
center_row, center_col = rows // 2, cols // 2  # 이미지의 중심 좌표
size = 30  # 통과시킬 저주파 영역의 크기 (n x n)
# 가운데 = 1, 외곽 = 0
mask = np.zeros((rows, cols, 2), np.uint8)
mask[center_row - size:center_row + size, center_col - size:center_col + size] = 1
lpf = dft_shift * mask  # 마스크를 적용한다. (요소별 곱셈: n x 1 = n, n x 0 = 0)
lpf_spectrum = 20 * np.log(1 + cv2.magnitude(lpf[:, :, 0], lpf[:, :, 1]))

""" LPF가 적용된 역 이산 푸리에 변환(LPF with IDFT) """
dft_ishift2 = np.fft.ifftshift(lpf)  # 셔플링 되었던 것을 역셔플링한다.
lpf_idft = cv2.idft(dft_ishift2)  # 역 이산 푸리에 변환을 한다.
lpf_idft = cv2.magnitude(lpf_idft[:, :, 0], lpf_idft[:, :, 1])  # 절대값 적용

# 결과물을 출력한다.
plt.subplot(231), plt.imshow(img, cmap='gray')  # 원본 영상
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(232), plt.imshow(spectrum, cmap='gray')  # DFT 영상
plt.title('DFT'), plt.xticks([]), plt.yticks([])

plt.subplot(233), plt.imshow(idft, cmap='gray')  # IDFT 영상
plt.title('IDFT'), plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(lpf_idft, cmap='gray')  # LPF가 적용된 IDFT 영상
plt.title('IDFT(LPF apply)'), plt.xticks([]), plt.yticks([])

plt.subplot(235), plt.imshow(lpf_spectrum, cmap='gray')  # LPF가 적용된 IDFT 영상
plt.title('DFT(LPF apply)'), plt.xticks([]), plt.yticks([])
plt.show()
