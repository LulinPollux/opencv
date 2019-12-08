import cv2
import numpy as np
from matplotlib import pyplot as plt

# 원본 이미지를 입력한다.
img = cv2.imread('../res/lenna.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


""" 고속 푸리에 변환(FFT) """
fft = np.fft.fft2(img)  # 고속 푸리에 변환을 적용한다.
fft_shift = np.fft.fftshift(fft)  # 분석을 쉽게 하기 위해 셔플링을 한다.
spectrum = 20 * np.log(1 + np.abs(fft_shift))  # 스펙트럼 영상을 구한다. D(u, v) = c x log( |F(u, v)| )

""" 역방향 고속 푸리에 변환(IFFT) """
hpf_ishift = np.fft.ifftshift(fft_shift)  # 셔플링 되었던 것을 역셔플링한다.
ifft = np.fft.ifft2(hpf_ishift)  # 역방향 고속 푸리에 변환을 한다.
ifft = np.abs(ifft)  # 절대값 적용

""" 고주파 통과 필터링(HPF) """
rows, cols = img.shape
center_row, center_col = rows // 2, cols // 2  # 이미지의 중심 좌표
size = 20  # 제거시킬 저주파 영역의 크기 (n x n)
# 가운데 = 0, 외곽 = 1
hpf_mask = np.ones((rows, cols), np.uint8)
hpf_mask[center_row - size:center_row + size, center_col - size:center_col + size] = 0
hpf = fft_shift * hpf_mask  # 마스크를 적용한다. (요소별 곱셈: n x 1 = n, n x 0 = 0)
hpf_spectrum = 20 * np.log(1 + np.abs(hpf))

""" HPF가 적용된 역방향 고속 푸리에 변환(HPF with IFFT) """
hpf_ishift = np.fft.ifftshift(hpf)  # 셔플링 되었던 것을 역셔플링한다.
hpf_ifft = np.fft.ifft2(hpf_ishift)  # 역방향 고속 푸리에 변환을 한다.
hpf_ifft = np.abs(hpf_ifft)  # 절대값 적용


# 결과물을 출력한다.
plt.subplot(231), plt.imshow(img, cmap='gray')  # 원본 영상
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(232), plt.imshow(spectrum, cmap='gray')  # FFT 영상
plt.title('FFT'), plt.xticks([]), plt.yticks([])

plt.subplot(233), plt.imshow(ifft, cmap='gray')  # IFFT 영상
plt.title('IFFT'), plt.xticks([]), plt.yticks([])

plt.subplot(234), plt.imshow(hpf_ifft, cmap='gray')  # HPF가 적용된 IFFT 영상
plt.title('IFFT(HPF apply)'), plt.xticks([]), plt.yticks([])

plt.subplot(235), plt.imshow(hpf_spectrum, cmap='gray')  # HPF가 적용된 FFT 영상
plt.title('FFT(HPF apply)'), plt.xticks([]), plt.yticks([])

plt.subplot(236), plt.imshow(hpf_mask, cmap='gray')  # HPF 마스크
plt.title('HPF Mask'), plt.xticks([]), plt.yticks([])
plt.show()
