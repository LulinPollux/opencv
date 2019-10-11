import cv2

src = cv2.imread("../res/1.jpg", cv2.IMREAD_COLOR)
dst1 = cv2.flip(src, 0)     # 상하대칭
dst2 = cv2.flip(src, 1)     # 좌우대칭

cv2.imshow("src", src)
cv2.imshow("dst1", dst1)
cv2.imshow("dst2", dst2)
cv2.waitKey(0)
cv2.destroyAllWindows()
