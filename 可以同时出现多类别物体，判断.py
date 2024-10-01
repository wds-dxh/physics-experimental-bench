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

# 定义线的起点和终点坐标
start_point = (320, 0)
end_point = (320, 480)

# 定义线的颜色和线宽
color = (0, 255, 0)  # (B, G, R)
thickness = 2
#记录单摆运行周期的列表
result_list = []
Is_ball_n = 0#默认让单摆的预测位置在第一位
# 读取视频流的线程
def video_stream_thread():
    global current_frame
    # video_stream_url = 'http://172.20.10.4:5000/video_feed'
    video_stream_url = 'save_video/单摆.avi'
    cap = cv2.VideoCapture(video_stream_url)

    while cap.isOpened():
        success, frame = cap.read()
        #视频文件的情况下，延时等待推理线程
        time.sleep(0.03)
        if success:
            with frame_lock:
                current_frame = frame
        else:
            break
    cap.release()


# 推理和处理数据的线程
def inference_thread():
    global Is_ball_n
    global current_frame
    start_time = time.time()#计算帧率
    result_startime = time.time()   #记录单摆开始运行的时间
    while True:
        with frame_lock:
            if current_frame is not None:
                results = model.predict(current_frame, conf=0.6, imgsz=(640, 480), max_det=2,stream=True)
                # 在帧上可视化结果
                results = list(results) # 转换为列表
                annotated_frame = results[0].plot()
                if len(results[0].boxes) != 0:
                    need_result = results[0].boxes
                    need_class_box = need_result.xyxy  # 框
                    need_class_list = need_result.cls  # 类别
                    for i in range(len(need_class_list)):
                        if need_class_list[0].item() == 0:
                            Is_ball_n = i
                    # print(need_class_box)
                    class_x = need_class_box[Is_ball_n][0]
                    # class_y = need_class_box[0][1]
                    # class_w = need_class_box[0][2]
                    # class_h = need_class_box[0][3]
                    if class_x.item() < 310 and class_x.item() > 290:
                        print("单摆在中心")
                        result_list.append(time.time()-result_startime)
                        result_startime = time.time()
                        #如果记录的单摆运行周期大于10个，则退出程序
                        if len(result_list) > 10:
                            print("单摆运行周期：",result_list)
                            sys.exit(0)

                    # 画出中心线
                cv2.line(annotated_frame, start_point, end_point, color, thickness)
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(annotated_frame, f"{fps:.1f} FPS", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
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