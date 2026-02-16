
#!/bin/bash
# ============================================================
# One-time setup: Connect GitHub Actions to your GCP project
# Each team member runs this with their own GCP project
#
# This script handles:
# - Workload Identity Federation (GitHub → GCP auth)
# - All IAM permissions needed for the pipeline
# - Enables required GCP APIs
# - Prints GitHub Variables to configure
#
# PREREQUISITES:
#   1. gcloud CLI installed (https://cloud.google.com/sdk/docs/install)
#   2. Authenticated: gcloud auth login
#   3. A GCP project with billing enabled
#   4. Owner or Editor role on the GCP project
#
# HOW TO RUN:
#   1. Change GCP_PROJECT_ID below to your GCP project ID
#   2. Run from the repo root:
#        cd trail-link/pipelines
#        chmod +x setup_github_oidc.sh
#        ./setup_github_oidc.sh
#   3. Add the printed GitHub Variables to:
#        https://github.com/Aditya-sairam/trail-link/settings/variables/actions
#        Prefix with your name (e.g., SWARALI_GCP_PROJECT_ID)
#   4. Add your GitHub username to pipelines/env-mapping.json
# ============================================================

set -e

# ---- CHANGE THESE VALUES ----
GCP_PROJECT_ID="your-gcp-project-id"         # e.g., datapipeline-infra
GCP_SA_NAME="triallink-pipeline-sa"          # service account name
GITHUB_REPO="Aditya-sairam/trail-link"       # GitHub org/repo
# ---- END CONFIGURATION ----

echo "============================================"
echo "=== TrialLink Pipeline - GCP Setup ==="
echo "============================================"
echo "Project: $GCP_PROJECT_ID"
echo "Repo: $GITHUB_REPO"
echo ""

# Get project number
PROJECT_NUMBER=$(gcloud projects describe $GCP_PROJECT_ID --format="value(projectNumber)")
SA_EMAIL="${GCP_SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
echo "Project Number: $PROJECT_NUMBER"
echo "Service Account: $SA_EMAIL"
echo ""

# ============================================================
# 1. WORKLOAD IDENTITY FEDERATION
# ============================================================
echo "=== [1/4] Setting up Workload Identity Federation ==="

echo "Creating Workload Identity Pool..."
gcloud iam workload-identity-pools create "github-pool" \
  --project="$GCP_PROJECT_ID" \
  --location="global" \
  --display-name="GitHub Actions Pool" 2>/dev/null || echo "  Pool already exists"

echo "Creating OIDC Provider..."
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project="$GCP_PROJECT_ID" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="attribute.repository=='${GITHUB_REPO}'" \
  --issuer-uri="https://token.actions.githubusercontent.com" 2>/dev/null || echo "  Provider already exists"

echo "Granting WIF identity binding..."
gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL \
  --project="$GCP_PROJECT_ID" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/${GITHUB_REPO}" \
  --quiet 2>/dev/null || echo "  Already granted"

echo ""

# ============================================================
# 2. PIPELINE SERVICE ACCOUNT PERMISSIONS
# ============================================================
echo "=== [2/4] Granting Pipeline SA permissions ==="

# Already in Pulumi but adding here for completeness / manual setup
echo "  storage.objectAdmin (read/write GCS buckets)..."
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/storage.objectAdmin" --quiet 2>/dev/null || echo "  Already granted"

echo "  artifactregistry.reader (pull Docker images)..."
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/artifactregistry.reader" --quiet 2>/dev/null || echo "  Already granted"

echo "  artifactregistry.writer (push Docker images from GitHub Actions)..."
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/artifactregistry.writer" --quiet 2>/dev/null || echo "  Already granted"

echo "  logging.logWriter (write logs)..."
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/logging.logWriter" --quiet 2>/dev/null || echo "  Already granted"

echo "  compute.instanceAdmin.v1 (start/stop VMs)..."
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/compute.instanceAdmin.v1" --quiet 2>/dev/null || echo "  Already granted"

echo "  iam.serviceAccountUser (SA acts as itself for VM metadata)..."
gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL \
  --project="$GCP_PROJECT_ID" \
  --role="roles/iam.serviceAccountUser" \
  --member="serviceAccount:$SA_EMAIL" --quiet 2>/dev/null || echo "  Already granted"

echo ""

# ============================================================
# 3. CLOUD BUILD DEFAULT SA PERMISSIONS (for Cloud Functions)
# ============================================================
echo "=== [3/4] Granting Cloud Build SA permissions ==="

echo "  artifactregistry.reader (default compute SA)..."
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.reader" --quiet 2>/dev/null || echo "  Already granted"

echo "  artifactregistry.writer (Cloud Build SA)..."
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/artifactregistry.writer" --quiet 2>/dev/null || echo "  Already granted"

echo "  storage.objectAdmin (Cloud Build SA)..."
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
  --role="roles/storage.objectAdmin" --quiet 2>/dev/null || echo "  Already granted"

echo "  cloudbuild.builds.builder (default compute SA)..."
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.builder" --quiet 2>/dev/null || echo "  Already granted"

echo ""

# ============================================================
# 4. ENABLE REQUIRED GCP APIs
# ============================================================
echo "=== [4/4] Enabling required GCP APIs ==="

gcloud services enable \
  storage.googleapis.com \
  compute.googleapis.com \
  artifactregistry.googleapis.com \
  cloudfunctions.googleapis.com \
  cloudscheduler.googleapis.com \
  pubsub.googleapis.com \
  iam.googleapis.com \
  logging.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  --project=$GCP_PROJECT_ID

echo ""

# ============================================================
# OUTPUT: GitHub Variables to set
# ============================================================
echo "============================================"
echo "=== SETUP COMPLETE ==="
echo "============================================"
echo ""
echo "Add these as GitHub Repository Variables"
echo "(Settings → Variables → Actions)"
echo "Prefix with your name, e.g., SWARALI_GCP_PROJECT_ID"
echo ""
echo "  GCP_PROJECT_ID = $GCP_PROJECT_ID"
echo "  GCP_REGION = us-central1"
echo "  GCP_SA_EMAIL = $SA_EMAIL"
echo "  WIF_PROVIDER = projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
echo "  GCP_VM_NAME = triallink-pipeline-vm"
echo "  GCP_VM_ZONE = us-central1-a"
echo ""
echo "Also add your GitHub username to pipelines/env-mapping.json"
echo "  e.g., {\"YourGitHubUsername\": \"YOUR_PREFIX\"}"
echo ""
echo "=== Done! ==="