"""
Main pulumi entry point - This is where the stacks will be created and pushed to GCP
"""

import pulumi 
import pulumi_gcp as gcp 
from patientStack import PatientStack

gcp_config = pulumi.Config("gcp")
project_id = gcp_config.require("project")
region = gcp_config.get("region") or "us-central1"  

stack_name = pulumi.get_stack() 

patient_stack = PatientStack(
    name=stack_name,
    project_id=project_id,
    region=region
)

pulumi.export("deployment_summary",{
    "environment":stack_name,
    "project":project_id,
    "region":region,
    "firestore_db":patient_stack.firestore_db.name
})
