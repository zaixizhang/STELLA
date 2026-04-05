FROM python:3.12-slim

LABEL org.opencontainers.image.title="STELLA"
LABEL org.opencontainers.image.description="Self-Evolving Multimodal Agents for Biomedical Research"
LABEL org.opencontainers.image.source="https://github.com/zaixizhang/STELLA"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose Gradio default port
EXPOSE 7860

# Environment variables (override at runtime via -e or .env mount)
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "start_stella_web.py"]
