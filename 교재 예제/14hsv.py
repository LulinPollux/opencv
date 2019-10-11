import cv2


def main1():
    src = cv2.imread("../res/hsv.jpg", cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    cv2.imshow("h", h)
    cv2.imshow("s", s)
    cv2.imshow("v", v)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main2():
    src = cv2.imread("../res/hsv.jpg", cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    h = cv2.inRange(h, 8, 20)
    orange = cv2.bitwise_and(hsv, hsv, mask = h)
    orange = cv2.cvtColor(orange, cv2.COLOR_HSV2BGR)

    cv2.imshow("orange", orange)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main2()
