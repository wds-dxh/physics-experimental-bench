import time
import cv2
from ultralytics import YOLO
import threading
from flask import Flask, render_template, Response
########################################
"""
识别后在推流
"""
############################################
# 加载YOLOv8模型
model = YOLO('best.pt')

# 用于存储当前帧的全局变量
current_frame = None
annotated_frame = None
frame_lock = threading.Lock()
app = Flask(__name__)

# 读取视频流的线程
def video_stream_thread():
    global current_frame
    video_stream_url = 'http://172.20.10.4:5000/video_feed'
    cap = cv2.VideoCapture(video_stream_url)

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            with frame_lock:
                current_frame = frame

    cap.release()

# 推理和处理数据的线程
def inference_thread():
    global current_frame
    global annotated_frame
    start_time = time.time()

    while True:
        with frame_lock:
            if current_frame is not None:
                # 在当前帧上运行YOLOv8推理
                results = model.predict(current_frame, conf=0.6, imgsz=(640, 480), max_det=300, save=True)

                # 在帧上可视化结果
                annotated_frame = results[0].plot()
                # annotated_frame = results.plot()#渲染

                fps = 1.0 / (time.time() - start_time)
                cv2.putText(annotated_frame, f"{fps:.1f} FPS", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                start_time = time.time()

        # 如果按下'q'则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# 生成视频流的帧
def generate_frames():
    global annotated_frame
    while True:
        if annotated_frame is not None:
            # 将帧编码为JPEG格式
            frame = cv2.resize(annotated_frame, (1920, 1080))
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame = buffer.tobytes()
            # 使用生成器函数输出帧
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # 使用Response对象包装生成器函数，以便将其作为视频流输出
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    app.run(debug=False, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    # 创建并启动两个线程
    video_thread = threading.Thread(target=video_stream_thread)
    inference_thread = threading.Thread(target=inference_thread)

    video_thread.start()
    inference_thread.start()

    # inference_thread.join()
    # video_thread.join()

    # 启动 Flask 应用（使用 Gunicorn 作为服务器）
    run_flask()
