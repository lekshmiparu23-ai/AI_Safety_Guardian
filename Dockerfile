FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose port 7860 (Hugging Face standard)
EXPOSE 7860

# Run Streamlit on port 7860
CMD ["streamlit", "run", "dashboard.py", "--server.port=7860", "--server.address=0.0.0.0"]
