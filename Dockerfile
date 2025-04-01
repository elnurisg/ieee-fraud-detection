# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Prevent Python from writing pyc files to disc & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Set the PORT environment variable to 8080 (used by nginx config)
ENV PORT=8080

# Install system dependencies including gcc, build-essential, and nginx
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential nginx && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies and supervisor
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install supervisor

# Copy the rest of the application code to the container
COPY . .

ENV PYTHONPATH=/app

# Copy custom Nginx configuration into the container
COPY nginx.conf /etc/nginx/nginx.conf

# Expose only the reverse proxy port (nginx listens on 8080)
EXPOSE 8080

# Run supervisord as the container's entrypoint, which will start nginx, FastAPI, and Streamlit
CMD ["supervisord", "-c", "/app/supervisord.conf"]
