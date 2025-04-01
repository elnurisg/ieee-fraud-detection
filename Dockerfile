# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Prevent Python from writing pyc files to disc & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
apt-get install -y --no-install-recommends gcc build-essential && \
rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .

ENV PYTHONPATH=/app

# Expose port 8080
EXPOSE 8080

# Run the FastAPI application with uvicorn using proper module notation
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
