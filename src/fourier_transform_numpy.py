import cv2
import numpy as np
from matplotlib import pyplot as plt

# 원본 이미지를 입력한다.
img = cv2.imread('../res/lenna.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

""" 순방향 고속 푸리에 변환(FFT)
    적용을 하면 화면 좌측상단(0, 0)이 중심이고, 그곳에 저주파가 모여 있다.
    분석을 쉽게 하기 위해 셔플링을 하고 Log Scaling을 하여 매우 넓은 범위의 값을 축소시킨다. """
fft = np.fft.fft2(img)  # 고속 푸리에 변환을 적용한다.
fft_shift = np.fft.fftshift(fft)  # 분석을 쉽게 하기 위해 셔플링을 한다.
spectrum_img = 20 * np.log(np.abs(fft_shift))  # 스펙트럼 영상을 구한다. D(u, v) = c x log( |F(u, v)| )

""" 고주파 통과 필터링(HPF)
    중앙에서 10X10 크기의 사각형의 값을 1로 설정하여, 중앙의 저주파를 모두 제거(검게 만듬)
    저주파를 제거했으므로 배경이 사라지고 에지만 남게 된다. """
row, col = img.shape
center_row, center_col = row // 2, col // 2  # 이미지의 중심 좌표
size = 10  # 제거할 저주파 영역의 크기 (n x n)
fft_shift[center_row - size:center_row + size, center_col - size:center_col + size] = 1

""" 역방향 고속 푸리에 변환(IFFT)
    푸리에 변환결과를 다시 이미지로 변환한다. """
fft_ishift = np.fft.ifftshift(fft_shift)  # 셔플링 되었던 것을 역셔플링한다.
ifft = np.fft.ifft2(fft_ishift)  # 역방향 고속 푸리에 변환을 한다.
ifft = np.abs(ifft)  # 절대값 적용

# 임계 값으로 이진 영상을 만든다.
ret, binary_img = cv2.threshold(np.uint8(ifft), 27, 255, cv2.THRESH_BINARY_INV)

# 결과물을 출력한다.
plt.subplot(221), plt.imshow(img, cmap='gray')  # 원본 영상
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(spectrum_img, cmap='gray')  # FFT 영상
plt.title('FFT'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(ifft, cmap='gray')  # HPF가 적용된 IFFT 영상
plt.title('IFFT(HPF apply)'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(binary_img, cmap='gray')  # 이진화된 IFFT 영상
plt.title('IFFT(HPF binary)'), plt.xticks([]), plt.yticks([])
plt.show()
