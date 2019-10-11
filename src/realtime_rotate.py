import cv2


def nothing(x):
    pass


src = cv2.imread("../res/1.jpg", cv2.IMREAD_COLOR)
height, width, channel = src.shape

cv2.namedWindow('image')
cv2.createTrackbar('degree', 'image', 0, 360, nothing)
cv2.createTrackbar('reverse', 'image', 0, 1, nothing)

while True:
    degree = cv2.getTrackbarPos('degree', 'image')
    reverse = cv2.getTrackbarPos('reverse', 'image')
    if reverse == 1:
        degree = -degree

    matrix = cv2.getRotationMatrix2D((width/2, height/2), degree, 1)
    dst = cv2.warpAffine(src, matrix, (width, height))

    cv2.imshow("image", dst)
    if cv2.waitKey(20) > 0:
        break

cv2.destroyAllWindows()
