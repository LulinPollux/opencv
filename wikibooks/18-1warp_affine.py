import numpy as np
import cv2

src = cv2.imread("../res/book.jpg", cv2.IMREAD_COLOR)
height, width, channel = src.shape

srcPoint = np.array([[283, 228], [59, 424], [281, 600]], dtype=np.float32)
dstPoint = np.array([[0, 0], [0, height], [width, height]], dtype=np.float32)
matrix = cv2.getAffineTransform(srcPoint, dstPoint)

dst = cv2.warpAffine(src, matrix, (width, height))

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
