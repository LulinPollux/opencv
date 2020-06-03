import os
import cv2
import tqdm

# 매개변수 설정
video_path = '../../test.mp4'
save_folder = '../../extract'
os.makedirs(save_folder, exist_ok=True)

# Video 열기
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print('Video opening process is failed.')
    exit(1)

# Video의 정보 받아오기
total_frame = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cipher = len(str(total_frame))

# 각 프레임을 이미지로 저장
for _ in tqdm.tqdm(range(total_frame), desc='Save frames'):
    ret, frame = cap.read()
    current_frame = str(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    filename = '{}.jpg'.format(current_frame.zfill(cipher))

    cv2.imwrite(os.path.join(save_folder, filename), frame)
cap.release()
