#!/bin/bash

# ========================================
# Build and push Docker image to GCP Artifact Registry
# Creates the repo if it doesn't already exist
# ========================================

set -e  # Exit immediately if a command fails

# ===== CONFIG =====
PROJECT_ID="project-1e322b00-efe1-41f8-8d9"
REGION="us-central1"
REPO_NAME="patient-api-repo-dev"
IMAGE_NAME="patient-api"
TAG="latest"

# ===== DERIVE FULL IMAGE NAME =====
FULL_IMAGE_NAME="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG"
echo "Full image name: $FULL_IMAGE_NAME"

# ===== DETERMINE SCRIPT DIRECTORY =====
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOCKERFILE_PATH="$SCRIPT_DIR/Dockerfile"
CONTEXT_PATH="$SCRIPT_DIR"

echo "Dockerfile path: $DOCKERFILE_PATH"
echo "Build context path: $CONTEXT_PATH"

# ===== CHECK IF ARTIFACT REGISTRY REPO EXISTS =====
echo "Checking if Artifact Registry repo '$REPO_NAME' exists..."
if gcloud artifacts repositories describe "$REPO_NAME" \
    --location="$REGION" \
    --project="$PROJECT_ID" > /dev/null 2>&1; then
    echo "✅ Repo '$REPO_NAME' already exists — skipping creation."
else
    echo "Repo '$REPO_NAME' not found — creating it..."
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --project="$PROJECT_ID" \
        --description="Patient API Backend"
    echo "✅ Repo '$REPO_NAME' created successfully."
fi

# ===== AUTHENTICATE DOCKER WITH GCP =====
echo "Configuring Docker to use GCP credentials..."
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

# ===== BUILD DOCKER IMAGE =====
echo "Building Docker image..."
docker build \
  -f "$DOCKERFILE_PATH" \
  -t "$FULL_IMAGE_NAME" \
  "$CONTEXT_PATH"
echo "✅ Docker image built successfully."

# ===== PUSH IMAGE TO ARTIFACT REGISTRY =====
echo "Pushing Docker image to Artifact Registry..."
docker push "$FULL_IMAGE_NAME"
echo "✅ Image successfully pushed to Artifact Registry!"