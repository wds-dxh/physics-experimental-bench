from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)




def generate_frames():
    
    # 使用OpenCV捕获摄像头
    camera = cv2.VideoCapture(0)
    # camera.set(3, 640)
    # camera.set(4, 480)

    while True:
        # 读取摄像头帧
        success, frame = camera.read()


        if not success:
            break
        else:
            # 将帧编码为JPEG格式
            frame = cv2.resize(frame, (640, 480))
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame = buffer.tobytes()

        # 使用生成器函数输出帧
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # 使用Response对象包装生成器函数，以便将其作为视频流输出
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # app.run(debug=True)
    #让外部可以访问
    app.run(debug=True, host='0.0.0.0' ,port=5000)
