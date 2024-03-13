import time

import cv2
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('best-wuli.pt')

# 打开视频文件
# video_stream_url = 'http://192.168.137.1:5000/video_feed'
# video_stream_url = 'http://192.168.9.9:5000/video_feed'
# video_stream_url = 'http://192.168.1.36:5000/video_feed'
video_stream_url = 'http://172.20.10.2:5000/video_feed'
cap = cv2.VideoCapture(video_stream_url)
start_time = time.time()
# 遍历视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()

    if success:
        # 在该帧上运行YOLOv8推理
        results = model.predict(frame,conf=0.6,imgsz=(640,480),max_det=1,save=True)

        # 在帧上可视化结果
        annotated_frame = results[0].plot()

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"{fps:.1f} FPS", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # 显示带注释的帧
        cv2.imshow("YOLOv8推理", annotated_frame)
        # 显示FPS

        start_time = time.time()
        # 如果按下'q'则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束则中断循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()