import cv2

image_file = "../res/6.jpg"    # 입력 파일
cascade_file = "C:/Users/Lulin/Miniconda3/envs/opencv/Lib/site-packages" \
               "/cv2/data/haarcascade_frontalface_alt.xml"  # 캐스케이드 파일

# 이미지를 읽는다.
image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)

# 그레이스케일로 변환한다.
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 인식 특징 파일을 읽는다.
cascade = cv2.CascadeClassifier(cascade_file)

# 얼굴 인식을 실행한다.
faces = cascade.detectMultiScale(grayscale, 1.1, 1)

if len(faces) > 0:
    # 인식한 부분을 표시한다.
    print(faces)
    for face in faces:
        x, y, w, h = face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=5)

    # 파일로 출력한다.
    cv2.imwrite("faceDetect.jpg", image)
else:
    print("얼굴 없음.")
