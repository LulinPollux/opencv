import os
import cv2
import tqdm

# 매개변수 설정
image_dir = '../../extract'
video_path = '../../video.mp4'
fps = 30

# 이미지의 파일명 불러오기
image_names = os.listdir(image_dir)

# VideoWriter 객체 생성
image = cv2.imread(os.path.join(image_dir, image_names[0]))
width, height = image.shape[1], image.shape[0]
out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
if not out.isOpened():
    print('Creating VideoWriter is failed.')
    exit(1)

# 이미지를 불러와서 비디오로 만든다.
for image_name in tqdm.tqdm(image_names, desc='Make video'):
    image = cv2.imread(os.path.join(image_dir, image_name))
    out.write(image)
out.release()
