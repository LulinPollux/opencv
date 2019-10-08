import cv2

image = cv2.imread("res/1.jpg", cv2.IMREAD_UNCHANGED)
height, width, channel = image.shape
print(height, width, channel)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
