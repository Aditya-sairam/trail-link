import pulumi
import pulumi_gcp as gcp
from typing import Optional


class DataPipelineStack:
    def __init__(self, name: str, project_id: str, region: str = "us-central1", opts: Optional[pulumi.ResourceOptions] = None):
        self.name = name
        self.project_id = project_id
        self.region = region
        self.opts = opts or pulumi.ResourceOptions()

        self.pipeline_bucket = self._create_bucket()
        self.service_account = self._create_service_account()
        self._create_artifact_registry()
        self.airflow_service = self._create_airflow_cloudrun_service() or None
        self._keep_alive_ping_for_airflow()
        self._grant_storage_access()
        self._export_outputs()
        self._make_public()

    def _create_bucket(self) -> gcp.storage.Bucket:
        return gcp.storage.Bucket(
            f"{self.name}-pipeline-data",
            name=f"triallink-pipeline-data-{self.project_id}",
            location=self.region,
            uniform_bucket_level_access=True,
            versioning=gcp.storage.BucketVersioningArgs(enabled=True),
            opts=self.opts,
        )

    def _create_service_account(self) -> gcp.serviceaccount.Account:
        return gcp.serviceaccount.Account(
            f"{self.name}-pipeline-service-account",
            account_id=f"pipeline-sa-{self.name}",
            display_name=f"Data Pipeline Service Account ({self.name})",
            opts=self.opts,
        )
    def _grant_storage_access(self):
        gcp.projects.IAMMember(
            f"{self.name}-pipeline-storage-access",
            project=self.project_id,
            role="roles/storage.objectAdmin",
            member=pulumi.Output.concat("serviceAccount:", self.service_account.email),
            opts=pulumi.ResourceOptions(parent=self.service_account),
        )

    def _create_artifact_registry(self):
        return gcp.artifactregistry.Repository(
            "data-pipeline-artifact-repo",
            location=self.region,
            repository_id=f"data-pipeline-repo-{self.name}",
            format="DOCKER",
            project=self.project_id,
            opts=self.opts,
            
        )

    ###Airflow DAGs and data pipeline infra setup
    def _create_airflow_cloudrun_service(self):
        return gcp.cloudrunv2.Service(
            f"{self.name}-airflow-service",
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
                    "image": "us-central1-docker.pkg.dev/mlops-test-project-486922/data-pipeline-repo-dev/datapipeline-api:latest",
                     "ports": {
                        "container_port": 8081,
                        },
                    "resources": {
                        "limits": {
                            "memory": "8Gi",
                            "cpu": "2",
                        },
                    },
                     "env": [
                        {"name": "CLINICAL_TRIALS_BUCKET", "value": self.pipeline_bucket.name},
                        {"name": "GCP_PROJECT_ID", "value": self.project_id},
                        {"name": "GCP_REGION", "value": self.region}
                    ],
                }],
            },
            opts=pulumi.ResourceOptions(
                depends_on=[self.service_account],
            ),
        )
    
    def _keep_alive_ping_for_airflow(self):
        return gcp.cloudscheduler.Job(
            "airflow-keep-alive-ping",
            region=self.region,
            project= self.project_id,
            description="Keeps the Airflow service alive by pinging it every 5 minutes",
            schedule="*/5 * * * *",
            time_zone="UTC",
            http_target=gcp.cloudscheduler.JobHttpTargetArgs(
                http_method="GET",
                uri=pulumi.Output.concat(self.airflow_service.uri, "/health"),
            ),
        )
    
    def _make_public(self):
        """Make Cloud Run service publicly accessible"""
        self.airflow_service = gcp.cloudrunv2.ServiceIamMember(
            f"{self.name}-airflow-public-access",
            project=self.project_id,
            location=self.region,
            name=self.airflow_service.name,
            role="roles/run.invoker",
            member="allUsers",
            opts=pulumi.ResourceOptions(parent=self.airflow_service),
        )

    def _export_outputs(self):
        pulumi.export(f"{self.name}_pipeline_bucket", self.pipeline_bucket.name)
        pulumi.export(f"{self.name}_pipeline_bucket_url", self.pipeline_bucket.url)
        pulumi.export(f"{self.name}_pipeline_sa", self.service_account.email)