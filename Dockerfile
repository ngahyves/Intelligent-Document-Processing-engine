# --- STAGE 1: BUILDER (Compilation) ---
FROM python:3.11-slim AS builder
#Working dir
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
#Installing libraries
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: RUNTIME (Final light Image) ---
FROM python:3.11-slim
WORKDIR /app

# System dependencies for EasyOcr
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy libraries installed in the builder
COPY --from=builder /usr/local /usr/local

#Copy all the projects without .dockerignore elements
COPY . .

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Run all the code
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]