# --- Base image ---
FROM python:3.11-slim

# --- Install system dependencies ---
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    imagemagick \
 && rm -rf /var/lib/apt/lists/*

# --- Set working directory ---
WORKDIR /app

# --- Copy project files ---
COPY . .

# --- Install dependencies ---
RUN pip install --upgrade pip
RUN pip install -e . black ruff pytest pytest-cov uvicorn

# --- Expose port ---
EXPOSE 8000

# --- Start app ---
CMD ["uvicorn", "ai_translator.api:app", "--host", "0.0.0.0", "--port", "8000"]
