import os
import pulumi
import pulumi_gcp as gcp
from typing import Optional

# Custom metric types — must match what alert_function/main.py writes
METRIC_AVG_SCORE        = "custom.googleapis.com/triallink/rag_avg_score"
METRIC_NOT_ELIGIBLE_PCT = "custom.googleapis.com/triallink/not_eligible_percentage"
METRIC_BREACHES_COUNT   = "custom.googleapis.com/triallink/alert_breaches_count"


class ModelPipelineStack:
    def __init__(self, name: str, project_id: str, region: str = "us-central1", opts: Optional[pulumi.ResourceOptions] = None):
        self.name = name
        self.project_id = project_id
        self.region = region
        self.opts = opts or pulumi.ResourceOptions()
        # self.model = self._deploy_gemini_model()
        self.service_account      = self._create_service_account()
        self.function_bucket      = self._create_function_bucket()
        self.eval_bucket          = self._create_eval_bucket()
        self.source_archive       = gcp.storage.BucketObject(
            "rag-service-zip",
            bucket=self.function_bucket.name,
            source=pulumi.FileArchive("../../models")
        )
        self.alert_source_archive = gcp.storage.BucketObject(
            "alert-function-zip",
            bucket=self.function_bucket.name,
            source=pulumi.FileArchive("../../models/alert_function")
        )
        self._create_firestore()
        self.pub_sub_service         = self._create_pub_sub_for_patient_request()
        self.eval_notification_topic = self._create_eval_notification_topic()
        self.rag_service_function    = self._deploy_cloud_function() 
        # self.alert_function          = self._deploy_alert_function()
        # self._setup_gcs_notification()
        # self._setup_alert_policies()                                  
        self._make_function_public()
        self._grant_permissions()
        self._export_outputs()


    def _create_firestore(self) -> gcp.firestore.Database:
        return gcp.firestore.Database(
            f"dev-clinical-trials-suggestions-db",
            project=self.project_id,
            name="clinical-trials-suggestions-db",
            location_id=self.region,
            type="FIRESTORE_NATIVE",
            concurrency_mode="OPTIMISTIC",
            opts=self. opts,
        )
    
    def _create_service_account(self) -> gcp.serviceaccount.Account:
        return gcp.serviceaccount.Account(
            f"{self.name}-model-service-account",
            account_id=f"model-sa-{self.name}",
            display_name=f"Model Pipeline Service Account ({self.name})",
            opts=self.opts,
        )

    def _deploy_gemini_model(self):
        gcp.vertex.AiEndpointWithModelGardenDeployment("deploy",
              publisher_model_name="publishers/google/models/medgemma@medgemma-4b-it",
            location=self.region,
            model_config={
                "accept_eula": True,
            },
        )

    def _create_pub_sub_for_patient_request(self):
        return gcp.pubsub.Topic(
            "patient-rag-requests",
            name="clinical-trial-suggestions-request",
            project=self.project_id,
        )

    def _create_eval_notification_topic(self):
        return gcp.pubsub.Topic(
            f"{self.name}-eval-notifications",
            name=f"eval-results-notifications-{self.name}",
            project=self.project_id,
        )

    def _create_function_bucket(self):
        return gcp.storage.Bucket(
            "function-storage-bucket",
            location=self.region,
            project=self.project_id,
            uniform_bucket_level_access=True
        )

    def _create_eval_bucket(self):
        return gcp.storage.Bucket(
            f"{self.name}-eval-results-bucket",
            name=f"triallink-eval-results-{self.name}-{self.project_id}",
            location=self.region,
            project=self.project_id,
            uniform_bucket_level_access=True,
        )

    # def _setup_gcs_notification(self):
    #     gcp.storage.Notification(
    #         f"{self.name}-eval-gcs-notification",
    #         bucket=self.eval_bucket.name,
    #         payload_format="JSON_API_V1",
    #         topic=self.eval_notification_topic.id,
    #         event_types=["OBJECT_FINALIZE"],
    #         object_name_prefix="eval_results/",
    #         opts=pulumi.ResourceOptions(
    #             depends_on=[self.eval_bucket, self.eval_notification_topic]
    #         ),
    #     )

    def _deploy_cloud_function(self):
        """RAG service Cloud Function — always deployed"""
        return gcp.cloudfunctionsv2.Function(
            "rag-service-cloud-function",
            location=self.region,
            project=self.project_id,
            build_config={
                "runtime"    : "python311",
                "entry_point": "run_rag_pipeline",
                "source": {
                    "storage_source": {
                        "bucket": self.function_bucket.name,
                        "object": self.source_archive.name,
                    }
                },
            },
            service_config={
                "max_instance_count"   : 5,
                "min_instance_count"   : 0,
                "available_memory"     : "1Gi",
                "timeout_seconds"      : 300,
                "service_account_email": self.service_account.email,
                "environment_variables": {
                    "GCP_PROJECT_ID"        : self.project_id,
                    "GOOGLE_FUNCTION_SOURCE": "rag_service.py",
                    "EVAL_BUCKET"           : f"triallink-eval-results-{self.name}-{self.project_id}",
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

    def _deploy_alert_function(self):
        """Alert Cloud Function — triggered by GCS notification via Pub/Sub"""
        return gcp.cloudfunctionsv2.Function(
            "eval-alert-cloud-function",
            location=self.region,
            project=self.project_id,
            build_config={
                "runtime"    : "python311",
                "entry_point": "handle_eval_complete",
                "source": {
                    "storage_source": {
                        "bucket": self.function_bucket.name,
                        "object": self.alert_source_archive.name,
                    }
                },
            },
            service_config={
                "max_instance_count"   : 3,
                "min_instance_count"   : 0,
                "available_memory"     : "256M",
                "timeout_seconds"      : 60,
                "service_account_email": self.service_account.email,
                "environment_variables": {
                    "GCP_PROJECT_ID": self.project_id,
                    "ALERT_EMAIL"   : os.getenv("ALERT_EMAIL", "sairam676820@gmail.com"),
                }
            },
            event_trigger=gcp.cloudfunctionsv2.FunctionEventTriggerArgs(
                trigger_region=self.region,
                event_type="google.cloud.pubsub.topic.v1.messagePublished",
                pubsub_topic=self.eval_notification_topic.id,
                service_account_email=self.service_account.email,
                retry_policy="RETRY_POLICY_RETRY",
            )
        )

    def _setup_alert_policies(self):
        """
        Cloud Monitoring Alert Policies — no return value.
        Skipped silently if ALERT_EMAIL not set.
        """
        alert_email = os.getenv("ALERT_EMAIL", "")
        if not alert_email:
            return

        notification_channel = gcp.monitoring.NotificationChannel(
            f"{self.name}-email-notification-channel",
            project=self.project_id,
            display_name=f"TrialLink Alert Email ({self.name})",
            type="email",
            labels={"email_address": alert_email},
        )

        gcp.monitoring.AlertPolicy(
            f"{self.name}-rag-score-alert",
            project=self.project_id,
            display_name="[TrialLink] RAG Average Score Degraded",
            combiner="OR",
            conditions=[{
                "display_name": "avg_overall_score below 3.0",
                "condition_threshold": {
                    "filter"         : f'resource.type="global" AND metric.type="{METRIC_AVG_SCORE}"',
                    "comparison"     : "COMPARISON_LT",
                    "threshold_value": 3.0,
                    "duration"       : "0s",
                    "aggregations"   : [{"alignment_period": "600s", "per_series_aligner": "ALIGN_MEAN"}],
                }
            }],
            notification_channels=[notification_channel.id],
            alert_strategy={"auto_close": "1800s"},
            documentation={"content": "RAG quality degraded. Avg score dropped below 3.0/5. Check eval_results in GCS."},
        )

        gcp.monitoring.AlertPolicy(
            f"{self.name}-not-eligible-alert",
            project=self.project_id,
            display_name="[TrialLink] High NOT ELIGIBLE Rate",
            combiner="OR",
            conditions=[{
                "display_name": "not_eligible_percentage above 70%",
                "condition_threshold": {
                    "filter"         : f'resource.type="global" AND metric.type="{METRIC_NOT_ELIGIBLE_PCT}"',
                    "comparison"     : "COMPARISON_GT",
                    "threshold_value": 70.0,
                    "duration"       : "0s",
                    "aggregations"   : [{"alignment_period": "600s", "per_series_aligner": "ALIGN_MEAN"}],
                }
            }],
            notification_channels=[notification_channel.id],
            alert_strategy={"auto_close": "1800s"},
            documentation={"content": "Over 70% of trials marked NOT ELIGIBLE. Check Vector Search index."},
        )

        gcp.monitoring.AlertPolicy(
            f"{self.name}-breach-alert",
            project=self.project_id,
            display_name="[TrialLink] Evaluation Threshold Breach",
            combiner="OR",
            conditions=[{
                "display_name": "alert_breaches_count above 0",
                "condition_threshold": {
                    "filter"         : f'resource.type="global" AND metric.type="{METRIC_BREACHES_COUNT}"',
                    "comparison"     : "COMPARISON_GT",
                    "threshold_value": 0,
                    "duration"       : "0s",
                    "aggregations"   : [{"alignment_period": "600s", "per_series_aligner": "ALIGN_MEAN"}],
                }
            }],
            notification_channels=[notification_channel.id],
            alert_strategy={"auto_close": "1800s"},
            documentation={"content": "Threshold breached. Check eval_results/summary.json in GCS."},
        )

    def _make_function_public(self):
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
        roles = [
            "roles/datastore.user",
            "roles/aiplatform.user",
            "roles/discoveryengine.viewer",
            "roles/pubsub.publisher",
            "roles/pubsub.subscriber",
            "roles/storage.objectAdmin",
            "roles/monitoring.metricWriter",
        ]
        for role in roles:
            gcp.projects.IAMMember(
                f"{self.name}-model-sa-{role.split('/')[-1]}",
                project=self.project_id,
                role=role,
                member=pulumi.Output.concat("serviceAccount:", self.service_account.email),
                opts=pulumi.ResourceOptions(parent=self.service_account),
            )

        # gcp.pubsub.TopicIAMMember(
        #     f"{self.name}-gcs-pubsub-publisher",
        #     project=self.project_id,
        #     topic=self.eval_notification_topic.name,
        #     role="roles/pubsub.publisher",
        #     member=pulumi.Output.concat(
        #         "serviceAccount:service-",
        #         self.project_id,
        #         "@gs-project-accounts.iam.gserviceaccount.com"
        #     ),
        # )

    def _export_outputs(self):
        pulumi.export("RAG_SERVICE_FUNCTION",    self.rag_service_function.name)
        pulumi.export("RAG_PIPELINE_TOPIC",      self.pub_sub_service.name)
        pulumi.export("EVAL_BUCKET",             self.eval_bucket.name)
        # pulumi.export("EVAL_NOTIFICATION_TOPIC", self.eval_notification_topic.name)