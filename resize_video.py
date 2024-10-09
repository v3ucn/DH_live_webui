import cv2

videoCapture = cv2.VideoCapture('quiet_output/gakki.mp4')

fps = 30  # 保存视频的帧率
size = (720, 1280)  # 保存视频的大小

# 使用 H.264 编码器
fourcc = cv2.VideoWriter_fourcc(*'H264') #或者 cv2.VideoWriter_fourcc('H','2','6','4')
videoWriter = cv2.VideoWriter('gakki_new.mp4', fourcc, fps, size)
i = 0

while True:
    success, frame = videoCapture.read()
    if success:
        i += 1
        if (i >= 1 and i <= 8000):
            frame = cv2.resize(frame, (384, 288))
            videoWriter.write(frame)

        if (i > 8000):
            print("success resize")
            break
    else:
        print('end')
        break

videoWriter.release()
videoCapture.release()