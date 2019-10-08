import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cascade_file = "C:/Users/Lulin/Miniconda3/envs/opencv/Lib/site-packages" \
               "/cv2/data/haarcascade_frontalface_alt.xml"  # 캐스케이드 파일
cascade = cv2.CascadeClassifier(cascade_file)

while True:
    ret, frame = capture.read()
    grayscale = cv2.cvtColor(frame, cv2.IMREAD_GRAYSCALE)
    faces = cascade.detectMultiScale(grayscale, 1.1, 1)

    # 인식한 부분을 표시한다.
    for face in faces:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(1) > 0:
        break

capture.release()
cv2.destroyAllWindows()
