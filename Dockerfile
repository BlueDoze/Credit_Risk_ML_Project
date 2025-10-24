FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
# Install only production dependencies with optimized pip installation
RUN pip install --no-cache-dir -r requirements.txt \
    && find /usr/local/lib/python3.9/site-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.9/site-packages -name "__pycache__" -delete

# Copy project files
# Copy only necessary files and directories
COPY src/ /app/src/
COPY config/ /app/config/
COPY main.py /app/

# Create necessary directories
RUN mkdir -p models data logs

CMD ["python", "main.py"]