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
        self._grant_storage_access()
        self._export_outputs()

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

    def _export_outputs(self):
        pulumi.export(f"{self.name}_pipeline_bucket", self.pipeline_bucket.name)
        pulumi.export(f"{self.name}_pipeline_bucket_url", self.pipeline_bucket.url)
        pulumi.export(f"{self.name}_pipeline_sa", self.service_account.email)