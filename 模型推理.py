from ultralytics import YOLO

# 加载模型
model = YOLO('yolov8n.pt')  # 预训练的 YOLOv8n 模型

# 在图片列表上运行批量推理
results = model(['ultralytics/assets/bus.jpg'])  # 返回 Results 对象列表

# 处理结果列表
for result in results:
    boxes = result.boxes  # 边界框输出的 Boxes 对象
    masks = result.masks  # 分割掩码输出的 Masks 对象
    keypoints = result.keypoints  # 姿态输出的 Keypoints 对象
    probs = result.probs  # 分类输出的 Probs 对象