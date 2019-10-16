import cv2


def play_camera():
    # cap 이 정상적으로 open이 되었는지 확인하기 위해서 cap.isOpen() 으로 확인가능
    cap = cv2.VideoCapture(0)
    print(cap.isOpened())

    # cap.get(prodId)/cap.set(propId, value)을 통해서 속성 변경이 가능.
    # 3은 width, 4는 heigh
    print('width: %d, height: %d' % (cap.get(3), cap.get(4)))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cv2.waitKey(10) == -1:
        # ret: frame capture결과 (boolean)
        # frame: Capture한 frame
        ret, frame = cap.read()

        if ret:
            # image를 Grayscale로 변환한다.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', gray)

    cap.release()
    cv2.destroyAllWindows()


def play_file():
    cap = cv2.VideoCapture('../res/space.mp4')

    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)

        if cv2.waitKey(25) > 0:
            break
    cap.release()
    cv2.destroyAllWindows()


def save_video():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.avi', fourcc, 30, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) > 0:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
