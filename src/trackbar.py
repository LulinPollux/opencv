import numpy as np
import cv2 as cv


def nothing(x):
    pass


# Create a black image, a window
img = np.zeros((300, 512, 3), np.uint8)
cv.namedWindow('image')

# create trackbars for color change
cv.createTrackbar('R', 'image', 0, 255, nothing)
cv.createTrackbar('G', 'image', 0, 255, nothing)
cv.createTrackbar('B', 'image', 0, 255, nothing)

# create switch for ON/OFF functionality
switch = 'OFF/ON'
cv.createTrackbar(switch, 'image', 0, 1, nothing)

while True:
    # get current positions of four trackbars
    r = cv.getTrackbarPos('R', 'image')
    g = cv.getTrackbarPos('G', 'image')
    b = cv.getTrackbarPos('B', 'image')
    s = cv.getTrackbarPos(switch, 'image')

    # 스위치가 꺼져 있으면 흑백, 켜져 있으면 색상
    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]
    
    # 이미지 표시
    cv.imshow('image', img)
    if cv.waitKey(10) > 0:
        break
cv.destroyAllWindows()
