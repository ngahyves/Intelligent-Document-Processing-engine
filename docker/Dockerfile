# 1. Base light image
FROM python:3.11-slim

# 2. Working directory
WORKDIR /app

# 3. Installing system dependencies (Required for OpenCV/EasyOCR)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4.  Python dependencies installation
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy of source code and models
COPY . .

# 6. Environment variable to help Python locate the modules
ENV PYTHONPATH=/app

# 7. Port 
EXPOSE 8000

# 8. Command to run the api
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]