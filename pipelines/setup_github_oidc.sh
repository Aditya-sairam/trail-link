#!/bin/bash
set -e

# ---- CHANGE THIS VALUE ----
GCP_PROJECT_ID="datapipeline-infra"
GCP_SA_NAME="triallink-pipeline-sa"
GITHUB_REPO="Aditya-sairam/trail-link"
# ---- END CONFIGURATION ----

echo "=== Setting up Workload Identity Federation ==="
echo "Project: $GCP_PROJECT_ID"
echo "Repo: $GITHUB_REPO"

PROJECT_NUMBER=$(gcloud projects describe $GCP_PROJECT_ID --format="value(projectNumber)")
echo "Project Number: $PROJECT_NUMBER"

echo "=== Creating Workload Identity Pool ==="
gcloud iam workload-identity-pools create "github-pool" \
  --project="$GCP_PROJECT_ID" \
  --location="global" \
  --display-name="GitHub Actions Pool" 2>/dev/null || echo "Pool already exists"

echo "=== Creating OIDC Provider ==="
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project="$GCP_PROJECT_ID" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com" 2>/dev/null || echo "Provider already exists"

echo "=== Granting IAM binding ==="
SA_EMAIL="${GCP_SA_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts add-iam-policy-binding $SA_EMAIL \
  --project="$GCP_PROJECT_ID" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/${GITHUB_REPO}"

echo ""
echo "============================================"
echo "=== SET THESE AS GITHUB VARIABLES ==="
echo "============================================"
echo ""
echo "GCP_PROJECT_ID = $GCP_PROJECT_ID"
echo "GCP_REGION = us-central1"
echo "GCP_SA_EMAIL = $SA_EMAIL"
echo "WIF_PROVIDER = projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
echo "GCP_VM_NAME = triallink-pipeline-vm"
echo "GCP_VM_ZONE = us-central1-a"