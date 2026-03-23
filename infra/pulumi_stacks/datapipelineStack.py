import pulumi
import pulumi_gcp as gcp
from typing import Optional
from pulumi_command import local


class DataPipelineStack:
    def __init__(self, name: str, project_id: str, region: str = "us-central1", opts: Optional[pulumi.ResourceOptions] = None):
        self.name = name
        self.project_id = project_id
        self.region = region
        self.opts = opts or pulumi.ResourceOptions()
        self.firestore_db = self._create_firestore()
        self.pipeline_bucket = self._create_bucket()
        self.service_account = self._create_service_account()
        self._create_artifact_registry()
        self.vector_index = self._create_vector_search_index()       
        self.vector_endpoint = self._create_vector_search_endpoint() 
        # self._deploy_index_to_endpoint() 
        self.airflow_service = self._create_airflow_cloudrun_service() or None
        self._keep_alive_ping_for_airflow()
        
        self._grant_storage_access()
        self._grant_vertex_ai_access()  
        self._grant_access_to_firestore()    
        self._make_public()

        self._export_outputs()

        self.dvc_bucket = gcp.storage.Bucket(
            f"dvc-storage-clinical-trials",
            name=f"dvc-storage-clinical-trials",
            location="US",
            versioning=gcp.storage.BucketVersioningArgs(
                enabled=True,
            ),
            force_destroy=True
        )

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

    def _grant_vertex_ai_access(self):
        """Grant Vertex AI permissions so Airflow can embed + query Vector Search"""
        gcp.projects.IAMMember(
            f"{self.name}-pipeline-vertex-ai-access",
            project=self.project_id,
            role="roles/aiplatform.user",
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
                        {"name": "GCP_REGION", "value": self.region},
                        {"name":"VECTOR_SEARCH_INDEX_ID","value":self.vector_index.id},
                        {"name":"FIRESTORE_DB","value":self.firestore_db.name}
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
            project=self.project_id,
            description="Keeps the Airflow service alive by pinging it every 5 minutes",
            schedule="50 23 * * 6",
            time_zone="UTC",
            http_target=gcp.cloudscheduler.JobHttpTargetArgs(
                http_method="GET",
                uri=pulumi.Output.concat(self.airflow_service.uri, "/health"),
            ),
        )

    def _create_firestore(self) -> gcp.firestore.Database:
        return gcp.firestore.Database(
            f"{self.name}-clinical-trials-db",
            project=self.project_id,
            name="clinical-trials-db",
            location_id=self.region,
            type="FIRESTORE_NATIVE",
            concurrency_mode="OPTIMISTIC",
            opts=self.opts,
        )
    def _grant_access_to_firestore(self):
        """Grant Vertex AI permissions so Airflow can embed + query Vector Search"""
        gcp.projects.IAMMember(
            f"{self.name}-pipeline-firestore-access",
            project=self.project_id,
            role="roles/datastore.owner",
            member=pulumi.Output.concat("serviceAccount:", self.service_account.email),
            opts=pulumi.ResourceOptions(parent=self.service_account),
        )

    def _create_vector_search_index(self) -> gcp.vertex.AiIndex:
        """
        Creates the Vertex AI Vector Search index for clinical trials.
        - 768 dimensions to match text-embedding-005
        - STREAM_UPDATE enables the streaming upsert used in embed.py
        - COSINE_DISTANCE matches the similarity metric used for patient matching
        """
        return gcp.vertex.AiIndex(
            f"{self.name}-trials-vector-index",
            project=self.project_id,
            region=self.region,
            display_name=f"clinical-trials-index-{self.name}",
            metadata=gcp.vertex.AiIndexMetadataArgs(
                contents_delta_uri="",
                config=gcp.vertex.AiIndexMetadataConfigArgs(
                    dimensions=768,
                    approximate_neighbors_count=10,
                    distance_measure_type="DOT_PRODUCT_DISTANCE",
                    feature_norm_type="UNIT_L2_NORM",
                    algorithm_config=gcp.vertex.AiIndexMetadataConfigAlgorithmConfigArgs(
                        tree_ah_config=gcp.vertex.AiIndexMetadataConfigAlgorithmConfigTreeAhConfigArgs(
                            leaf_node_embedding_count=500,
                            leaf_nodes_to_search_percent=7,
                        )
                    ),
                ),
            ),
            index_update_method="STREAM_UPDATE",
            opts=pulumi.ResourceOptions(
                parent=self.firestore_db,
                depends_on=[self.firestore_db],
            ),
        )

    def _create_vector_search_endpoint(self) -> gcp.vertex.AiIndexEndpoint:
        """
        Creates the endpoint that the index is deployed to.
        This is what the patient API will query for trial matching.
        """
        return gcp.vertex.AiIndexEndpoint(
            f"{self.name}-trials-index-endpoint",
            project=self.project_id,
            region=self.region,
            display_name=f"clinical-trials-endpoint-{self.name}",
            opts=pulumi.ResourceOptions(
                depends_on=[self.vector_index],
            ),
        )

    def _deploy_index_to_endpoint(self):
        gcp.vertex.AiIndexEndpointDeployedIndex(
            f"{self.name}-deployed-trials-index",
            region=self.region,
            index_endpoint=self.vector_endpoint.id,
            index=self.vector_index.id,
            deployed_index_id=f"clinical_trials_{self.name}",
            display_name=f"clinical-trials-{self.name}",
            opts=pulumi.ResourceOptions(
                depends_on=[self.vector_endpoint, self.vector_index],
                retain_on_delete=True,      # ← never delete real resource
                ignore_changes=["*"],       # ← never try to update it
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
        pulumi.export("RAW_CLINICAL_TRIALS_STORAGE", self.pipeline_bucket.name)
        pulumi.export("CLINICAL_TRIALS_FIRESTORE", self.firestore_db.name)
        pulumi.export(f"{self.name}_pipeline_sa", self.service_account.email)
        pulumi.export(f"{self.name}_vector_search_index_id", self.vector_index.id)
        pulumi.export(f"{self.name}_vector_search_endpoint_id", self.vector_endpoint.id)
        # pulumi.export(f"{self.name}_medgemma_endpoint_id", self.medgemma_endpoint.id)

    