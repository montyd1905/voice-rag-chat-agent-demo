FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libpq-dev \
    gcc \
    ffmpeg \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies with BuildKit cache mount for persistence
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install --cache-dir=/root/.cache/pip -r requirements.txt || \
    (echo "Retrying pip install..." && sleep 5 && pip install --cache-dir=/root/.cache/pip -r requirements.txt) || \
    (echo "Final retry..." && sleep 10 && pip install --cache-dir=/root/.cache/pip -r requirements.txt)

# Download spaCy English model with cache mount and retry logic
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --cache-dir=/root/.cache/pip --default-timeout=300 https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl || \
    (echo "First retry (30s delay)..." && sleep 30 && pip install --cache-dir=/root/.cache/pip --default-timeout=300 https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl) || \
    (echo "Second retry (60s delay)..." && sleep 60 && pip install --cache-dir=/root/.cache/pip --default-timeout=300 https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl)

# Copy application code (this layer changes most frequently)
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

