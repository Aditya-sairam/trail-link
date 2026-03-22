import pulumi
import pulumi_gcp as gcp
from typing import Optional


class ModelPipelineStack:
    def __init__(self, name: str, project_id: str, region: str = "us-central1", opts: Optional[pulumi.ResourceOptions] = None):
        self.name = name
        self.project_id = project_id
        self.region = region
        self.opts = opts or pulumi.ResourceOptions()
        self.clinical_trial_suggestions_store = self.clinical_trials_firestore()
        self.service_account = self._create_service_account()
        self.function_bucket = self._funnction_bucket()
        self.source_achive = gcp.storage.BucketObject(
            "rag-service-zip",
            bucket=self.function_bucket.name,
            source=pulumi.FileArchive("../../models")
        )
        self.pub_sub_service = self._create_pub_sub_for_patient_request()
        self.rag_service_function = self._deploy_cloud_function()
       
        # self._deploy_gemini_model()
        self._make_function_public()   # ← new
        self._grant_permissions()
        self._export_outputs()

    def _create_service_account(self) -> gcp.serviceaccount.Account:
        return gcp.serviceaccount.Account(
            f"{self.name}-model-service-account",
            account_id=f"model-sa-{self.name}",
            display_name=f"Model Pipeline Service Account ({self.name})",
            opts=self.opts,
        )

    def _create_pub_sub_for_patient_request(self):
        return gcp.pubsub.Topic(
            "patient-rag-requests",
            name="clinical-trial-suggestions-request",
            project=self.project_id,
        )

    def _funnction_bucket(self):
        return gcp.storage.Bucket(
            "function-storaage-bucket",
            location=self.region,
            project=self.project_id
        )

    def _deploy_cloud_function(self):
        return gcp.cloudfunctionsv2.Function(
            "rag-service-cloud-function",
            location=self.region,
            project=self.project_id,
            build_config={
                "runtime": "python311",
                "entry_point": "run_rag_pipeline",
                "source": {
                    "storage_source": {
                        "bucket": self.function_bucket.name,
                        "object": self.source_achive.name,
                    }
                },
            },
            service_config={
                "max_instance_count": 5,
                "min_instance_count": 0,
                "available_memory": "1Gi",
                "timeout_seconds": 300,
                "service_account_email": self.service_account.email,
                "environment_variables": {
                    "GCP_PROJECT_ID": self.project_id,
                    "MODEL_PROJECT_ID": self.project_id,
                    "MEDGEMMA_ENDPOINT_ID": "mg-endpoint-bb15ba35-9f1b-4101-acda-037a1c2d3de0",
                    "GOOGLE_FUNCTION_SOURCE": "rag_service.py",
                    "TRAIL_SUGGESTIONS_STORE":self.clinical_trial_suggestions_store.name
                }
            },
            event_trigger=gcp.cloudfunctionsv2.FunctionEventTriggerArgs(
                trigger_region=self.region,
                event_type="google.cloud.pubsub.topic.v1.messagePublished",
                pubsub_topic=self.pub_sub_service.id,
                service_account_email=self.service_account.email,
                retry_policy="RETRY_POLICY_RETRY",
            )
        )

    def _make_function_public(self):
        """Allow unauthenticated invocations of the Cloud Function"""
        gcp.cloudfunctionsv2.FunctionIamMember(
            f"{self.name}-rag-function-public",
            project=self.project_id,
            location=self.region,
            cloud_function=self.rag_service_function.name,
            role="roles/cloudfunctions.invoker",
            member="allUsers",
            opts=pulumi.ResourceOptions(parent=self.rag_service_function),
        )

    def _grant_permissions(self):
        for role in [
            "roles/datastore.user",
            "roles/aiplatform.user",
            "roles/discoveryengine.viewer",
            "roles/pubsub.publisher",
            "roles/pubsub.subscriber",
        ]:
            gcp.projects.IAMMember(
                f"{self.name}-model-sa-{role.split('/')[-1]}",
                project=self.project_id,
                role=role,
                member=pulumi.Output.concat("serviceAccount:", self.service_account.email),
                opts=pulumi.ResourceOptions(parent=self.service_account),
            )

    def clinical_trials_firestore(self):
        return  gcp.firestore.Database(
            f"{self.name}-clinical-trials-suggestions-db",
            project=self.project_id,
            name="clinical-trials-suggestions-db",
            location_id=self.region,
            type="FIRESTORE_NATIVE",
            concurrency_mode="OPTIMISTIC",
            opts=self.opts,
            )
    def _export_outputs(self):
        pulumi.export("RAG_SERVICE_FUNCTION", self.rag_service_function.name)
        pulumi.export("RAG_PIPELINE_TOPIC", self.pub_sub_service.name)