FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the YOLO11x-pose model
RUN python -c "from ultralytics import YOLO; YOLO('yolo11x-pose.pt')"

COPY . .

ENV YOLO_CONFIG_DIR=/tmp

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
