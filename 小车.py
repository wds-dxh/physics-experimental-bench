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

# 读取视频流的线程
def video_stream_thread():
    global current_frame
    video_stream_url = 'http://172.20.10.2:5000/video_feed'
    cap = cv2.VideoCapture(video_stream_url)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while cap.isOpened():
        success, frame = cap.read()
        time.sleep(0.03)
        if success:
            with frame_lock:
                current_frame = frame
        else:
            break

    cap.release()

# 推理和处理数据的线程
def inference_thread():
    global current_frame
    start_time = time.time()
    signal_print_start = 0
    while True:
        with frame_lock:
            if cv2.waitKey(1) & 0xFF == ord("s"):
                result_startime = time.time()
                if signal_print_start == 0:
                    signal_print_start = 1
                    print("小车开始运行时刻：", result_startime)

            if cv2.waitKey(1) & 0xFF == ord("d"):
                result_endtime = time.time()
                print("小车到达终点时刻：", result_endtime)
                print("小车运行时间：", result_endtime - result_startime)
                print("前半段时间：", result_middletime - result_startime)
                print("后半段时间：", result_endtime - result_middletime)
                write_excel.write_data_to_excel("experimental_data/car.xlsx",result_middletime - result_startime,result_endtime - result_middletime)
                # write_excel.write_data_to_excel("expedddddddddddrimental_data/car.xlsx", 0.2, 0.3)
                sys.exit(0)  # 退出程序
            if current_frame is not None:
                # 在当前帧上运行YOLOv8推理
                results = model.predict(current_frame, conf=0.6, imgsz=(640, 480), max_det=1)

                # 在帧上可视化结果
                annotated_frame = results[0].plot()
                if len(results[0].boxes) != 0:
                    need_result = results[0].boxes
                    need_class_box = need_result.xyxy  # 框
                    # print(need_class_box)
                    class_x = need_class_box[0][0]
                    # class_y = need_class_box[0][1]
                    # class_w = need_class_box[0][2]
                    # class_h = need_class_box[0][3]
                    if class_x.item() < 310 and class_x.item() > 290:
                        print("小车在中心的时刻：", time.time())
                        result_middletime = time.time()

                    # 画出中心线
                cv2.line(annotated_frame, start_point, end_point, color, thickness)

                fps = 1.0 / (time.time() - start_time)
                cv2.putText(annotated_frame, f"{fps:.1f} FPS", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 显示带注释的帧
                cv2.imshow("YOLOv8推理", annotated_frame)

                start_time = time.time()

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