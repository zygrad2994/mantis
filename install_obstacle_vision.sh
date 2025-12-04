#!/bin/bash
set -e

echo "=== Обновляем систему ==="
sudo apt update && sudo apt upgrade -y

echo "=== Устанавливаем системные пакеты ==="
sudo apt install -y build-essential cmake pkg-config git python3-pip \
  python3-venv python3-dev ffmpeg gstreamer1.0-tools gstreamer1.0-libav \
  libgl1-mesa-dev mesa-utils libxkbcommon-x11-0 libzmq3-dev pigpio \
  librealsense2-utils librealsense2-dev

echo "=== Создаём каталог проекта ==="
sudo mkdir -p /opt/tactical-ar
cd /opt/tactical-ar

echo "=== Создаём виртуальное окружение ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Устанавливаем Python-зависимости ==="
cat > requirements.txt << 'EOF'
pyzmq
PyQt5
numpy
opencv-python
pyopengl
pyopengl-accelerate
psutil
scipy
rclpy
sensor-msgs
geometry-msgs
hailo-sdk-client
pyrealsense2
ydlidar
EOF

pip install -r requirements.txt

echo "=== Копируем основной файл системы ==="
cat > dahua_ar_vision_system.py << 'EOF'
#!/usr/bin/env python3
import sys, cv2, numpy as np, rclpy, time, os, threading, math, psutil, subprocess
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from hailo_sdk_client import InferenceContext
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont

# Цветовые схемы зон
ZONE_COLORS = {
    'safe': (0, 255, 0),
    'attention': (255, 255, 0),
    'danger': (255, 0, 0),
    'unknown': (128, 128, 128)
}

class DahuaVisionSystem(Node):
    def __init__(self):
        super().__init__('dahua_obstacle_system')
        self.dahua_active = False
        self.lidar_active = False
        self.hailo_active = False
        self.current_frame = None
        self.lidar_ranges = None
        self.lidar_angles = None
        self.camera_fov = 80.0
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0.0
        self.initialize_sensors()
        self.lidar_subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def initialize_sensors(self):
        threading.Thread(target=self.initialize_dahua, daemon=True).start()
        try:
            self.hailo_context = InferenceContext()
            self.hef_path = os.path.expanduser("~/.hailo/models/yolov8n_obstacle.hef")
            if os.path.exists(self.hef_path):
                self.hailo_context.add_hef_file(self.hef_path)
                self.hailo_active = True
                self.get_logger().info("✅ Hailo-8 успешно инициализирован (YOLOv8n Obstacle)")
            else:
                self.get_logger().error(f"❌ Файл модели не найден: {self.hef_path}")
        except Exception as e:
            self.get_logger().error(f"❌ Ошибка инициализации Hailo: {str(e)}")

    def initialize_dahua(self):
        rtsp_url = "rtsp://admin:Admin12345@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
        self.dahua_cap = cv2.VideoCapture(rtsp_url)
        if self.dahua_cap.isOpened():
            self.dahua_active = True
            self.get_logger().info("✅ Dahua камера подключена")

    def lidar_callback(self, msg):
        self.lidar_ranges = np.array(msg.ranges)
        self.lidar_ranges[np.isinf(self.lidar_ranges)] = 0.0
        self.lidar_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        self.lidar_active = True

    def get_lidar_distance(self, pixel_x, frame_width):
        if not self.lidar_active or self.lidar_ranges is None or self.lidar_angles is None:
            return float('inf')
        pixel_angle = (pixel_x - frame_width/2) * (self.camera_fov / frame_width)
        pixel_angle_rad = math.radians(pixel_angle)
        diffs = np.abs(self.lidar_angles - pixel_angle_rad)
        idx = np.argmin(diffs)
        return self.lidar_ranges[idx] if self.lidar_ranges[idx] > 0 else float('inf')

    def preprocess_frame(self, frame):
        input_size = 640
        resized = cv2.resize(frame, (input_size, input_size))
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=0)

    def postprocess_yolo(self, outputs, img_shape):
        detections = []
        if outputs is None or len(outputs) == 0:
            return detections
        arr = outputs.reshape(-1, 5)  # [x1,y1,x2,y2,conf]
        for x1,y1,x2,y2,conf in arr:
            if conf < 0.75: continue
            detections.append([int(x1), int(y1), int(x2), int(y2), float(conf)])
        return detections

    def process_frame_step(self):
        if not self.dahua_active: return None, []
        ret, frame = self.dahua_cap.read()
        if not ret: return None, []
        current_frame = cv2.resize(frame, (640, 480))
        detections = []
        if self.hailo_active:
            preprocessed = self.preprocess_frame(current_frame)
            outputs = self.hailo_context.run(preprocessed)  # реальный инференс
            detections = self.postprocess_yolo(outputs, current_frame.shape)
        fused_objects = []
        for det in detections:
            x1,y1,x2,y2,conf = det
            center_x = int((x1+x2)/2)
            distance = self.get_lidar_distance(center_x, current_frame.shape[1])
            status = "safe"
            if distance < 0.3: status = "danger"
            elif distance < 0.5: status = "attention"
            elif distance == float('inf'): status = "unknown"
            fused_objects.append({
                'bbox': (x1,y1,x2,y2),
                'cls': "obstacle",
                'confidence': conf,
                'distance': distance,
                'status': status
            })
        # FPS расчёт
        self.frame_count += 1
        now = time.time()
        if now - self.last_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_time)
            self.frame_count = 0
            self.last_time = now
        return current_frame, fused_objects

    def get_system_stats(self):
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        try:
            temp_out = subprocess.check_output(["vcgencmd","measure_temp"]).decode()
            temp = float(temp_out.split('=')[1].split("'")[0])
        except Exception:
            temp = 0.0
        return cpu, ram, temp, self.fps

class DahuaVisionGUI(QMainWindow):
    def __init__(self, vision_node):
        super().__init__()
        self.vision_node = vision_node
        self.setWindowTitle("Obstacle Vision System")
        self.setGeometry(100,100,1280,720)
        central = QWidget(); self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self.ar_label = QLabel(); layout.addWidget(self.ar_label)
        self.timer = QTimer(); self.timer.timeout.connect(self.update_display); self.timer.start(50)

    def update_display(self):
        frame, objects = self.vision_node.process_frame_step()
        if frame is None: return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        qt_img = QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        painter = QPainter(pixmap); painter.setRenderHint(QPainter.Antialiasing)
        font = QFont("Arial",12,QFont.Bold); painter.setFont(font)

        # HUD системные показатели
        cpu, ram, temp, fps = self.vision_node.get_system_stats()
        hud_text = f"FPS: {fps:.1f} | CPU: {cpu:.1f}% | RAM: {ram:.1f}% | TEMP: {temp:.1f}°C"
        painter.setPen(QPen(QColor(255,255,255),1))
        painter.drawText(QRectF(10,10,400,20), Qt.AlignLeft, hud_text)

        # Отображение объектов
        for obj in objects:
            x1,y1,x2,y2 = obj['bbox']
            dist = obj['distance']; status = obj['status']
            color = QColor(*ZONE_COLORS[status])
            painter.setPen(QPen(color,2))
            painter.drawRect(x1,y1,x2-x1,y2-y1)
            text = f"obstacle {dist:.2f}m {status.upper()}"
            painter.drawText(QRectF(x1,y1-20,200,20),Qt.AlignLeft,text)

        painter.end()
        self.ar_label.setPixmap(pixmap.scaled(self.ar_label.width(),self.ar_label.height(),
                                              Qt.KeepAspectRatio,Qt.SmoothTransformation))

def main(args=None):
    rclpy.init(args=args)
    vision_node = DahuaVisionSystem()
    app = QApplication(sys.argv)
    gui = DahuaVisionGUI(vision_node)
    gui.show()
    app.exec_()
    vision_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
EOF

echo "=== Создаём systemd-сервис ==="
cat > tactical-ar.service << 'EOF'
[Unit]
Description=Obstacle Vision System (Dahua + LIDAR + Hailo-8 + RealSense)
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/tactical-ar
Environment=PYTHONUNBUFFERED=1
ExecStart=/opt/tactical-ar/venv/bin/python3 /opt/tactical-ar/dahua_ar_vision_system.py
Restart=always
RestartSec=5
WatchdogSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo cp tactical-ar.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tactical-ar
sudo systemctl start tactical-ar

echo "=== Настройка pigpio для аудио-сигналов ==="
sudo systemctl enable pigpiod
sudo systemctl start pigpiod

echo "=== Установка завершена! Система будет запускаться автоматически при старте Raspberry Pi 5 ==="
