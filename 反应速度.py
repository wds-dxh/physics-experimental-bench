import sys
import time
import cv2
import write_excel
import threading
import os

os.environ['YOLO_VERBOSE'] = str(False)
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('best.pt')

# 用于存储当前帧的全局变量
current_frame = None
frame_lock = threading.Lock()
countdown_num = 0
countdown_time = None
countdown_num1 = 0

# 读取视频流的线程
def video_stream_thread():
    global current_frame
    video_stream_url = 'http://172.20.10.4:5000/video_feed'
    # video_stream_url = 'save_video/单摆.avi'
    cap = cv2.VideoCapture(video_stream_url)

    while cap.isOpened():
        success, frame = cap.read()
        #视频文件的情况下，延时等待推理线程
        # time.sleep(0.03)
        if success:
            with frame_lock:
                current_frame = frame
        else:
            break
    cap.release()


# 推理和处理数据的线程
def inference_thread():
    global countdown_num1
    global countdown_num
    global current_frame
    start_time = time.time()#计算帧率
    while True:
        with frame_lock:
            if current_frame is not None:
                results = model.predict(current_frame, conf=0.6, imgsz=(640, 480), max_det=1)
                # 在帧上可视化结果
                # results = list(results) # 转换为列表
                for result in results:
                    annotated_frame = result.plot()
                    if len(result.boxes) != 0 & countdown_num == 2:
                        need_class_dit = result.names  # 字典类别
                        print(need_class_dit)
                        for i in range(len(need_class_dit)):
                            if list(need_class_dit.keys())[i] == 2:
                                print("反应速度",time.time() - result_startime)
                                sys.exit()

                countdown_num1 += 1
                if countdown_num1%50 == 1:
                    if countdown_num < 3:
                        countdown_num += 1;
                        print(countdown_num)
                        # cv2.putText(annotated_frame, f"{int(countdown_num):d} countdown", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if countdown_num == 3:
                    result_startime = time.time()
                if countdown_num < 4:
                    cv2.putText(annotated_frame, f"{int(countdown_num):d} countdown", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(annotated_frame, f"{fps:.1f} FPS", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #org:意思是将文本写到图片的哪个位置，font:字体，fontScale:字体大小，color:颜色，thickness:字体粗细
                start_time = time.time()
                # 显示带注释的帧
                cv2.imshow("YOLOv8推理", annotated_frame)


        # 如果按下'q'则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# 创建并启动两个线程
video_thread = threading.Thread(target=video_stream_thread)
inference_thread = threading.Thread(target=inference_thread)

video_thread.start()
inference_thread.start()

# 等待两个线程结束
video_thread.join()
inference_thread.join()

# 关闭所有窗口
cv2.destroyAllWindows()