import cv2
import numpy as np
from matplotlib import pyplot as plt
from pyplot_template import plt_arch

# 원본 이미지를 입력한다.
img = cv2.imread('../res/lenna.bmp', cv2.IMREAD_GRAYSCALE)


""" 고속 푸리에 변환(FFT) """
fft = np.fft.fft2(img)  # 고속 푸리에 변환을 적용한다.
fft_shift = np.fft.fftshift(fft)  # 분석을 쉽게 하기 위해 셔플링을 한다.
spectrum = 20 * np.log(1 + np.abs(fft_shift))  # 스펙트럼 영상을 구한다. D(u, v) = c x log( |F(u, v)| )

""" 역방향 고속 푸리에 변환(IFFT) """
hpf_ishift = np.fft.ifftshift(fft_shift)  # 셔플링 되었던 것을 역셔플링한다.
ifft = np.fft.ifft2(hpf_ishift)  # 역방향 고속 푸리에 변환을 한다.
ifft = np.abs(ifft)  # 절대값 적용

""" 저주파 통과 필터링(LPF) """
rows, cols = img.shape
center_row, center_col = rows // 2, cols // 2  # 이미지의 중심 좌표
size = 30  # 통과시킬 저주파 영역의 크기 (n x n)
# 가운데 = 1, 외곽 = 0
lpf_mask = np.zeros((rows, cols), np.uint8)
cv2.circle(lpf_mask, (center_row, center_col), size, (255, 255, 255), cv2.FILLED)
lpf = fft_shift * lpf_mask  # 마스크를 적용한다. (요소별 곱셈: n x 1 = n, n x 0 = 0)
lpf_spectrum = 20 * np.log(1 + np.abs(lpf))

""" 고주파 통과 필터링(HPF) """
rows, cols = img.shape
center_row, center_col = rows // 2, cols // 2  # 이미지의 중심 좌표
size = 30  # 제거시킬 저주파 영역의 크기 (n x n)
# 가운데 = 0, 외곽 = 1
hpf_mask = np.ones((rows, cols), np.uint8)
cv2.circle(hpf_mask, (center_row, center_col), size, (0, 0, 0), cv2.FILLED)
hpf = fft_shift * hpf_mask  # 마스크를 적용한다. (요소별 곱셈: n x 1 = n, n x 0 = 0)
hpf_spectrum = 20 * np.log(1 + np.abs(hpf))

""" LPF가 적용된 역 이산 푸리에 변환(LPF with IDFT) """
lpf_ishift = np.fft.ifftshift(lpf)  # 셔플링 되었던 것을 역셔플링한다.
lpf_ifft = np.fft.ifft2(lpf_ishift)  # 역 이산 푸리에 변환을 한다.
lpf_ifft = np.abs(lpf_ifft)  # 절대값 적용

""" HPF가 적용된 역방향 고속 푸리에 변환(HPF with IFFT) """
hpf_ishift = np.fft.ifftshift(hpf)  # 셔플링 되었던 것을 역셔플링한다.
hpf_ifft = np.fft.ifft2(hpf_ishift)  # 역방향 고속 푸리에 변환을 한다.
hpf_ifft = np.abs(hpf_ifft)  # 절대값 적용


# 결과물을 출력한다.
plt_arch(331, 'Input Image', img)
plt_arch(332, 'FFT', spectrum)
plt_arch(333, 'IFFT', ifft)
plt_arch(334, 'IFFT(LPF apply)', lpf_ifft)
plt_arch(335, 'FFT(HPF apply)', lpf_spectrum)
plt_arch(336, 'LPF Mask', lpf_mask)
plt_arch(337, 'IFFT(HPF apply)', hpf_ifft)
plt_arch(338, 'FFT(HPF apply)', hpf_spectrum)
plt_arch(339, 'HPF Mask', hpf_mask)
plt.show()
