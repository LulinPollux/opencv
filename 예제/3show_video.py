import cv2

capture = cv2.VideoCapture("res/space.mp4")

while True:
    ret, frame = capture.read()
    cv2.imshow("Video", frame)

    if cv2.waitKey(33) > 0:
        break

capture.release()
cv2.destroyAllWindows()
