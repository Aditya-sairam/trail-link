#!/bin/bash

# ========================================
# Build and push Docker image to GCP Artifact Registry
# Workaround for Pulumi Docker build issues
# ========================================

set -e  # Exit immediately if a command fails

# ===== CONFIG =====
PROJECT_ID="mlops-test-project-486922"
REGION="us-central1"
REPO_NAME="data-pipeline-repo-dev"
IMAGE_NAME="datapipeline-api"
TAG="latest"

# ===== DERIVE FULL IMAGE NAME =====
FULL_IMAGE_NAME="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$TAG"
echo "Full image name: $FULL_IMAGE_NAME"

# ===== DETERMINE SCRIPT DIRECTORY =====
# This makes the script runnable from anywhere
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

DOCKERFILE_PATH="$SCRIPT_DIR/Dockerfile"
CONTEXT_PATH="$SCRIPT_DIR"

echo "Dockerfile path: $DOCKERFILE_PATH"
echo "Build context path: $CONTEXT_PATH"

# ===== AUTHENTICATE DOCKER WITH GCP =====
echo "Configuring Docker to use GCP credentials..."
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet

# ===== BUILD DOCKER IMAGE =====
echo "Building Docker image..."
docker build \
  -f "$DOCKERFILE_PATH" \
  -t "$FULL_IMAGE_NAME" \
  "$CONTEXT_PATH"

echo "Docker image built successfully."

# ===== PUSH IMAGE TO ARTIFACT REGISTRY =====
echo "Pushing Docker image to Artifact Registry..."
docker push "$FULL_IMAGE_NAME"

echo "✅ Image successfully pushed to Artifact Registry!"
