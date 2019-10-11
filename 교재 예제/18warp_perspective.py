import numpy as np
import cv2

src = cv2.imread("../res/2.jpg", cv2.IMREAD_COLOR)
height, width, channel = src.shape

srcPoint = np.array([[300, 200], [400, 200], [500, 500], [200, 500]], dtype=np.float32)
dstPoint = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
matrix = cv2.getPerspectiveTransform(srcPoint, dstPoint)

dst = cv2.warpPerspective(src, matrix, (width, height))

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
