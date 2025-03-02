FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model_serving/flask /app
COPY models /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    SKLEARN_MODEL_PATH=/app/models/random_forest_latest.pkl \
    BERT_MODEL_PATH=/app/models/bert/bert_classifier_latest

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"] 