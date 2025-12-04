#!/bin/bash
set -e

echo "=== Обновляем систему ==="
sudo apt update && sudo apt upgrade -y

echo "=== Устанавливаем системные пакеты ==="
sudo apt install -y build-essential cmake pkg-config git python3-pip \
  python3-venv python3-dev ffmpeg gstreamer1.0-tools gstreamer1.0-libav \
  libgl1-mesa-dev mesa-utils libxkbcommon-x11-0 libzmq3-dev

# Pigpio: ставим Python-биндинги или собираем из исходников
sudo apt install -y python3-pigpio || {
  echo "⚠️ pigpio не найден в apt, собираем из исходников"
  git clone https://github.com/joan2937/pigpio.git
  cd pigpio && make -j4 && sudo make install && cd ..
  sudo systemctl enable pigpiod
  sudo systemctl start pigpiod
}

# RealSense: добавляем официальный репозиторий Intel
if ! dpkg -s librealsense2-utils >/dev/null 2>&1; then
  echo "=== Устанавливаем Intel RealSense SDK ==="
  sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6B0FC61
  sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo $(lsb_release -cs) main"
  sudo apt update
  sudo apt install -y librealsense2-utils librealsense2-dev librealsense2-dkms
fi

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
# сюда вставь полный код Obstacle Only версии с HUD
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
