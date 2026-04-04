import pulumi
import pulumi_gcp as gcp
from typing import Optional


class ModelDeployStack:
    def __init__(
        self,
        name: str,
        project_id: str,
        region: str = "us-central1",
        opts: Optional[pulumi.ResourceOptions] = None,
    ):
        self.name = name
        self.project_id = project_id
        self.region = region
        self.opts = opts or pulumi.ResourceOptions()

        self.endpoint = self._deploy_gemini_model()
        self._export_outputs()

    def _deploy_gemini_model(self):
        return gcp.vertex.AiEndpointWithModelGardenDeployment(
            "deploy",
            publisher_model_name="publishers/google/models/medgemma@medgemma-4b-it",
            location=self.region,
            model_config={
                "accept_eula": True,
            },
        )

    def _export_outputs(self):
        pulumi.export("MEDGEMMA_ENDPOINT_ID", self.endpoint.id)