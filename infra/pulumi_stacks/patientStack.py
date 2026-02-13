"""
In Infra coding, its better to have a set of reated resources to be defined within the same stack.
In this code, we habve all the patient related infra setup under the same stack.
"""
import pulumi
import pulumi_gcp as gcp
from pulumi import Config, export
from typing import Optional
import pulumi_docker as docker
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


class PatientStack:
    def __init__(self,name:str,project_id:str,region:str="us-central1",opts:Optional[pulumi.ResourceOptions]=None):
        self.name = name 
        self.project_id = project_id 
        self.region = region 
        self.opts = opts or pulumi.ResourceOptions()

        self.firestore_db = self._create_firestore()
        self.artifact_repo = self._create_artifact_repiository()
        self.api_image = "us-central1-docker.pkg.dev/mlops-test-project-486922/patient-api-repo-dev/patient-api:latest"
        self.service_account = self._create_service_account()
        self._grant_permissions()
        self.api_service = self._create_cloud_run()
        self._export_outputs()
        self._make_public()
        
    def _create_firestore(self) -> gcp.firestore.Database:
        return gcp.firestore.Database(
            f"{self.name}-patient-database",
            project = self.project_id,
            name=f"patient-db-{self.name}",
            location_id = self.region,
            type = "FIRESTORE_NATIVE",
            concurrency_mode = "OPTIMISTIC",
            opts = self.opts
        )
    
    def _create_artifact_repiository(self) -> gcp.artifactregistry.Repository:
        return gcp.artifactregistry.Repository("patient-api-artifact-repo",
                                               location = self.region,
                                               repository_id = f"patient-api-repo-{self.name}",
                                               format = "DOCKER",
                                               project = self.project_id)
     
    def _create_service_account(self) -> gcp.serviceaccount.Account:
        return gcp.serviceaccount.Account(
            f"{self.name}-api-service-account",
            account_id=f"patient-api-{self.name}",
            display_name=f"Patient API Service Account ({self.name})",
            opts=self.opts,
        )
    
    def _grant_permissions(self):
        gcp.firestore_role = gcp.projects.IAMMember(
            f"{self.name}-api-firestore-access",
            project = self.project_id,
            role = "roles/datastore.user",
            member = pulumi.Output.concat("serviceAccount:",self.service_account.email),
            opts=pulumi.ResourceOptions(parent=self.service_account),
        )

    def _create_cloud_run(self):
        return gcp.cloudrunv2.Service(
            f"{self.name}-patient-api-service-new",
            location=self.region,
            project=self.project_id,
            ingress="INGRESS_TRAFFIC_ALL",
            template={
                "health_check_disabled": False,  
                "service_account": self.service_account.email,
                "scaling": {
                    "min_instance_count": 0,
                    "max_instance_count": 5,
                },
                "containers": [{
                    "image": self.api_image,
                     "ports": {
                        "container_port": 8080,
                        },
                    "resources": {
                        "limits": {
                            "memory": "512Mi",
                            "cpu": "1",
                        },
                    },
                    "env": [
                        {"name": "ENVIRONMENT", "value": self.name},
                        {"name": "GCP_PROJECT_ID", "value": self.project_id},
                        {"name": "GCP_REGION", "value": self.region},
                        {"name": "FIRESTORE_DATABASE", "value": self.firestore_db.name}
                    ],
                }],
            },
            opts=pulumi.ResourceOptions(
                depends_on=[self.firestore_db, self.service_account],
            ),
        )


    
    def _make_public(self):
        """Make Cloud Run service publicly accessible"""
        self.api_iam_binding = gcp.cloudrunv2.ServiceIamMember(
            f"{self.name}-api-public-access",
            project=self.project_id,
            location=self.region,
            name=self.api_service.name,
            role="roles/run.invoker",
            member="allUsers",
            opts=pulumi.ResourceOptions(parent=self.api_service),
        )

    def _export_outputs(self):
        pulumi.export(f"{self.name}_firestore_db",self.firestore_db.name)


