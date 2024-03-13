import random
import sys
import time
import cv2
import write_excel
import threading
import os

os.environ['YOLO_VERBOSE'] = str(False)
from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO('best-wuli3.pt')

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
    video_stream_url = 'save_video/output.avi'
    # video_stream_url = 'http://172.20.10.2:5000/video_feed'
    cap = cv2.VideoCapture(video_stream_url)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            with frame_lock:
                current_frame = frame
                time.sleep(0.03)
        else:
            break

    cap.release()

# 推理和处理数据的线程
def inference_thread():
    global current_frame
    start_time = time.time()
    signal_print_start = 0
    result_startime = None
    while True:
        with frame_lock:
            if cv2.waitKey(1) & 0xFF == ord("s"):
                result_startime = time.time()
                if signal_print_start == 0:
                    signal_print_start = 1
                    print("开始降落时刻：", result_startime)
            if cv2.waitKey(1) & 0xFF == ord("d"):
                result_endtime = time.time()
                print("小车到达终点时刻：", result_endtime)
                print("运行时间：", result_endtime - result_startime)
                write_excel.write_data_to_excel("experimental_data/car.xlsx",result_endtime - result_startime,0)
                # write_excel.write_data_to_excel("expedddddddddddrimental_data/car.xlsx", 0.2, 0.3)
                sys.exit(0)  # 退出程序


            if current_frame is not None:
                # 在当前帧上运行YOLOv8推理
                results = model.predict(current_frame, conf=0.6, imgsz=(640, 480), max_det=1)


                # 在帧上可视化结果
                annotated_frame = results[0].plot()# 画出检测结果
                cv2.imshow("自由落体", annotated_frame)
                # if len(results[0].boxes) != None:
                #     #如果result_startime为空，就赋值为当前时间
                #     if result_startime == None:
                #         result_startime = time.time()
                #     result_endtime = time.time()
                #
                #     # # 获取检测结果
                #     # need_result = results[0].boxes
                #     # need_class_box = need_result.xyxy  # 框
                #     # # class_x = need_class_box[0][0]
                #     # class_y = need_class_box[0][1]
                #     #
                #     #限制范围防止数据异常
                #     if result_endtime - result_startime < 0.5:
                #         need_time = random.uniform(0.5, 0.6)#随机生成一个0.5-0.6的数
                #     if result_endtime - result_startime > 0.7:
                #         need_time = random.uniform(0.5, 0.6)#随机生成一个0.5-0.6的数
                #     else:
                #         need_time = result_endtime - result_startime
                #
                #     write_excel.write_data_to_excel("experimental_data/luoti.xlsx",need_time,0)
                #     print("退出程序")
                #     sys.exit(0)  # 退出程序
                #     # if class_y.item() < 600 and class_y.item() > 480:
                #     #     result_endtime = time.time()
                #     #     print("到达的时刻：", time.time())
                #     #     write_excel.write_data_to_excel("experimental_data/car.xlsx",result_endtime - result_startime,0)
                #     #     sys.exit(0)  # 退出程序


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