# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the frontend application code to the container
COPY . .

# Expose port 8080 (Cloud Run/Docker Compose internal port)
EXPOSE 8080

# Run the Streamlit app on port 8080 and bind to all network interfaces
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
