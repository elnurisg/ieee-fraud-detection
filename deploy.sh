#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables
PROJECT_ID="ieee-fraud-detection-455423"
SERVICE_NAME="fraud-detection-api"
REGION="us-central1"
IMAGE="gcr.io/$PROJECT_ID/$SERVICE_NAME"

# Step 1: Authenticate with Google Cloud
echo "Authenticating with Google Cloud..."
gcloud auth login
gcloud config set project $PROJECT_ID

# Step 2: Enable required APIs
echo "Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Step 3: Build the Docker image
echo "Building the Docker image..."
docker build -t $IMAGE .

# Step 4: Push the Docker image to Google Container Registry
echo "Pushing the Docker image to Google Container Registry..."
docker push $IMAGE

# Step 5: Deploy the application to Google Cloud Run
echo "Deploying the application to Google Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image=$IMAGE \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated \
    --timeout=600

# Step 6: Print the service URL
echo "Deployment completed successfully!"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')
echo "Your service is running at: $SERVICE_URL"

