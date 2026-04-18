"""
Main pulumi entry point - This is where the stacks will be created and pushed to GCP
"""

import pulumi
import pulumi_gcp as gcp
from patientStack import PatientStack
from datapipelineStack import DataPipelineStack
from modelPipelineStack import ModelPipelineStack
from modelDeployStack import ModelDeployStack

gcp_config  = pulumi.Config("gcp")
app_config  = pulumi.Config()

project_id  = gcp_config.require("project")
region      = gcp_config.get("region") or "us-central1"
stack_name  = pulumi.get_stack()

deploy_model    = app_config.get_bool("deploy_model")    or False
deploy_pipeline = app_config.get_bool("deploy_pipeline") or False
deploy_patient  = app_config.get_bool("deploy_patient")  or False

# ── Patient Stack ─────────────────────────────────────────────────────────────
patient_stack = None
if deploy_patient:
    patient_stack = PatientStack(
        name=stack_name,
        project_id=project_id,
        region=region
    )

# ── Data Pipeline Stack ───────────────────────────────────────────────────────
pipeline_stack = None
if deploy_pipeline:
    pipeline_stack = DataPipelineStack(
        name=stack_name,
        project_id=project_id,
        region=region,
    )

# ── Model Pipeline Stack (Cloud Function, Pub/Sub, Buckets) ──────────────────
model_stack = None
if deploy_pipeline:
    model_stack = ModelPipelineStack(
        name=stack_name,
        project_id=project_id,
        region=region
    )

# ── Model Deploy Stack (MedGemma — separate account) ─────────────────────────
model_deploy_stack = None
if deploy_model:
    model_deploy_stack = ModelDeployStack(
        name=stack_name,
        project_id=project_id,
        region=region
    )

# ── Exports ───────────────────────────────────────────────────────────────────
pulumi.export("deployment_summary", {
    "environment" : stack_name,
    "project"     : project_id,
    "region"      : region,
    "firestore_db": patient_stack.firestore_db.name if patient_stack else "not deployed",
})

if model_deploy_stack:
    pulumi.export("MEDGEMMA_ENDPOINT_ID", model_deploy_stack.endpoint.id)