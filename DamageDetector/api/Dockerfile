FROM python:3.12-slim

# Sending ultralytics' settings file to a folder that is writable inside the container, and turning off its habit of running pip install while the server is live.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    YOLO_AUTOINSTALL=false

# OpenCV, which ultralytics depends on, needs these two system libraries that the slim image leaves out.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installing the CPU only build of PyTorch first. The default build bundles gigabytes of NVIDIA GPU libraries that Cloud Run cannot use.
RUN pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu

# Copying the requirements before the code so this layer is reused when only main.py changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY weights ./weights

# Cloud Run tells the container which port to listen on through the PORT variable.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
