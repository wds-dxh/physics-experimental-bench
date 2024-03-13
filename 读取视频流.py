import cv2

# 视频流的地址，替换成你的Flask应用地址
video_stream_url = 'http://192.168.137.1:5000/video_feed'

# 打开视频流
cap = cv2.VideoCapture(video_stream_url)

while True:
    # 读取视频流帧
    ret, frame = cap.read()
    if not ret:
        break

    # 显示帧
    cv2.imshow('Video Stream', frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频流并关闭窗口
cap.release()
cv2.destroyAllWindows()
