FROM python:3.12-slim

WORKDIR /app

# Clean apt cache
RUN apt-get update && rm -rf /var/lib/apt/lists/*

# Install Python dependencies and docker CLI
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get update && \
    apt-get install -y docker.io && \
    rm -rf /var/lib/apt/lists/*

# Copy application files
COPY server.py .
COPY training.py .
COPY data_processor.py .
COPY templates/ templates/

# Create directories
RUN mkdir -p outputs training_data checkpoints

# Expose port
EXPOSE 5000

# Run the server
CMD ["python", "server.py"]

