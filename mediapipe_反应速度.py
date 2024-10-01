import sys
import threading

import cv2
import mediapipe as mp
import math
import HandTrackingModule as htm
import time

import write_excel

# detector = htm.handDetector(detectionCon=0.7)
# countdown_num = 0
# countdown_num1 = 0


current_frame = None
frame_lock = threading.Lock()
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

def inference_thread():
    detector = htm.handDetector(detectionCon=0.7)
    global current_frame
    countdown_num = 0
    countdown_num1 = 0
    cont_result_startime = 0
    while True:
        if current_frame is not None:
            img = current_frame
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)

            if len(lmList) != 0:
                # print(lmList[4], lmList[8])
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                length = math.hypot(x2 - x1, y2 - y1)  # 计算两点之间的距离
                if length < 15 :
                    if countdown_num == 3:
                        result_time= time.time() - result_startime-0.1
                        write_excel.write_data_to_excel("experimental_data/hand.xlsx", result_time," ")
                        print("反应速度", result_time)
                        break
                        sys.exit(0)

            countdown_num1 += 1
            if countdown_num1 % 50 == 1:
                if countdown_num < 3:
                    countdown_num += 1;
                    print(countdown_num)
            if countdown_num == 3:
                if cont_result_startime == 0:
                    result_startime = time.time()
                    cont_result_startime = 1

            if countdown_num < 4:
                cv2.putText(img, f"{int(countdown_num):d} countdown", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 255, 255), 5)

            # cTime = time.time()
            # fps = 1 / (cTime - pTime)
            # pTime = cTime
            # cv2.putText(img, f"{fps:.1f} FPS", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("IMG", img)
            key = cv2.waitKey(1) & 0xFF  # 按键处理
            if key == ord('q'):  # 当按下 'q' 键时退出循环
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