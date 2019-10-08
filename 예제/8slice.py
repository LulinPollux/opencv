import cv2


def main1():
    src = cv2.imread("res/2.jpg", cv2.IMREAD_COLOR)

    dst = src.copy()
    dst = src[100:600, 200:700]

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main2():
    src = cv2.imread("res/2.jpg", cv2.IMREAD_COLOR)

    dst = src.copy()
    slice_area = src[100:600, 200:700]
    dst[0:500, 0:500] = slice_area

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main2()
