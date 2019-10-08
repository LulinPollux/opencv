import datetime
import cv2

capture = cv2.VideoCapture("res/space.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False

while True:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    key = cv2.waitKey(33)

    if key == 27:   # ESC
        break
    elif key == 26:     # Ctrl + Z
        print("캡쳐")
        cv2.imwrite("capture" + str(now) + ".png", frame)
    elif key == 24:     # Ctrl + X
        print("녹화 시작")
        record = True
        video = cv2.VideoWriter("capture" + str(now) + ".avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    elif key == 3:      # Ctrl + C
        print("녹화 중지")
        record = False
        video.release()

    if record == True:
        print("녹화 중..")
        video.write(frame)

capture.release()
cv2.destroyAllWindows()
