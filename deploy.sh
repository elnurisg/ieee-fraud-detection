#!/bin/bash
set -e

# Define your project and region
PROJECT_ID="ieee-fraud-detection-455423"
REGION="us-central1"

# Backend service details
BACKEND_SERVICE="fraud-detection-backend"
BACKEND_IMAGE="gcr.io/$PROJECT_ID/$BACKEND_SERVICE"

# Frontend service details
FRONTEND_SERVICE="fraud-detection-frontend"
FRONTEND_IMAGE="gcr.io/$PROJECT_ID/$FRONTEND_SERVICE"

# Step 1: Authenticate with Google Cloud and set project
echo "Authenticating with Google Cloud..."
gcloud auth login
gcloud config set project $PROJECT_ID

# Step 2: Enable required APIs
echo "Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Step 3: Build and deploy the backend service
echo "Building backend Docker image..."
docker build -t $BACKEND_IMAGE -f Dockerfile .

echo "Pushing backend image..."
docker push $BACKEND_IMAGE

echo "Deploying backend to Cloud Run..."
gcloud run deploy $BACKEND_SERVICE \
    --image=$BACKEND_IMAGE \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated \
    --timeout=600

BACKEND_URL=$(gcloud run services describe $BACKEND_SERVICE --platform managed --region $REGION --format 'value(status.url)')
echo "Backend deployed at: $BACKEND_URL"

# Step 4: Build and deploy the frontend service
echo "Building frontend Docker image..."
docker build -t $FRONTEND_IMAGE -f app/frontend/Dockerfile app/frontend

echo "Pushing frontend image..."
docker push $FRONTEND_IMAGE

echo "Deploying frontend to Cloud Run..."
gcloud run deploy $FRONTEND_SERVICE \
    --image=$FRONTEND_IMAGE \
    --platform=managed \
    --region=$REGION \
    --allow-unauthenticated \
    --set-env-vars API_URL="${BACKEND_URL}/predict" \
    --timeout=600

FRONTEND_URL=$(gcloud run services describe $FRONTEND_SERVICE --platform managed --region $REGION --format 'value(status.url)')
echo "Frontend deployed at: $FRONTEND_URL"
