import pulumi
import pulumi_gcp as gcp

def create_pipeline_stack():
    config = pulumi.Config("gcp")
    project = config.require("project")
    region = config.require("region")

    # GCS Bucket for pipeline data and logs
    pipeline_bucket = gcp.storage.Bucket(
        "triallink-pipeline-data",
        name=f"triallink-pipeline-data-{project}",
        location=region,
        uniform_bucket_level_access=True,
        versioning=gcp.storage.BucketVersioningArgs(enabled=True),
    )

    # Artifact Registry for pipeline Docker images
    pipeline_registry = gcp.artifactregistry.Repository(
        "triallink-pipeline-registry",
        repository_id="triallink-pipeline",
        location=region,
        format="DOCKER",
        description="Docker images for TrialLink data pipeline",
    )

    # Service Account for the pipeline VM
    pipeline_sa = gcp.serviceaccount.Account(
        "pipeline-sa",
        account_id="triallink-pipeline-sa",
        display_name="TrialLink Data Pipeline Service Account",
        project=project,
    )

    # IAM: Allow SA to read/write GCS bucket
    gcp.projects.IAMMember(
        "pipeline-sa-storage",
        project=project,
        role="roles/storage.objectAdmin",
        member=pipeline_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )

    # IAM: Allow SA to pull Docker images from Artifact Registry
    gcp.projects.IAMMember(
        "pipeline-sa-artifact-reader",
        project=project,
        role="roles/artifactregistry.reader",
        member=pipeline_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )

    # IAM: Allow SA to write logs
    gcp.projects.IAMMember(
        "pipeline-sa-log-writer",
        project=project,
        role="roles/logging.logWriter",
        member=pipeline_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )

    # IAM: Allow SA to manage Compute Engine instances (for starting/stopping VM)
    gcp.projects.IAMMember(
        "pipeline-sa-compute-admin",
        project=project,
        role="roles/compute.instanceAdmin.v1",
        member=pipeline_sa.email.apply(lambda e: f"serviceAccount:{e}"),
    )

    # GCE VM for running the pipeline
    pipeline_vm = gcp.compute.Instance(
        "pipeline-vm",
        name="triallink-pipeline-vm",
        machine_type="e2-small",
        zone=f"{region}-a",
        boot_disk=gcp.compute.InstanceBootDiskArgs(
            initialize_params=gcp.compute.InstanceBootDiskInitializeParamsArgs(
                image="projects/cos-cloud/global/images/family/cos-stable",
                size=30,
            ),
        ),
        network_interfaces=[
            gcp.compute.InstanceNetworkInterfaceArgs(
                network="default",
                access_configs=[
                    gcp.compute.InstanceNetworkInterfaceAccessConfigArgs()
                ],
            ),
        ],
        service_account=gcp.compute.InstanceServiceAccountArgs(
            email=pipeline_sa.email,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        ),

    metadata_startup_script="""#!/bin/bash
set -x
LOG_FILE="/tmp/startup.log"
exec > >(tee $LOG_FILE) 2>&1

echo "=== STEP 1: VM Started at $(date) ==="
export HOME=/tmp

echo "=== STEP 2: Auth Docker ==="
docker-credential-gcr configure-docker --registries=us-central1-docker.pkg.dev

echo "=== STEP 3: Pull Image ==="
IMAGE="us-central1-docker.pkg.dev/datapipeline-infra/triallink-pipeline/triallink-pipeline:2026-02-16-042549"
docker pull $IMAGE

echo "=== STEP 4: Run Airflow DAG ==="
docker run --rm \
    -e AIRFLOW__CORE__LOAD_EXAMPLES=False \
    -e AIRFLOW__CORE__EXECUTOR=SequentialExecutor \
    -e AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////tmp/airflow.db \
    -e AIRFLOW_HOME=/tmp/airflow \
    -e GCS_BUCKET=triallink-pipeline-data-datapipeline-infra \
    $IMAGE \
    bash -c "mkdir -p /tmp/airflow && export AIRFLOW_HOME=/tmp/airflow && cp -r /opt/airflow/dags /tmp/airflow/dags && airflow db migrate && airflow dags test clinical_trials_pipeline 2026-02-16"

echo "=== DAG exit code: $? ==="

echo "=== STEP 5: Upload log to GCS ==="
TOKEN=$(curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | sed 's/.*"access_token":"\\([^"]*\\)".*/\\1/')
LOGNAME="startup-$(date -u +%Y%m%d-%H%M%S).log"
curl -sf -X POST \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: text/plain" \
    --data-binary @$LOG_FILE \
    "https://storage.googleapis.com/upload/storage/v1/b/triallink-pipeline-data-datapipeline-infra/o?uploadType=media&name=logs/$LOGNAME"

echo "=== STEP 6: Shutting down ==="
shutdown -h now
""",
#  metadata_startup_script="""#!/bin/bash
# set -x
# LOG_FILE="/tmp/startup.log"
# exec > $LOG_FILE 2>&1

# echo "=== STEP 1: VM Started at $(date) ==="
# echo "=== Running as user: $(whoami) ==="

# echo "=== STEP 2: Check Docker ==="
# which docker
# docker version

# echo "=== STEP 3: Auth Docker ==="
# docker-credential-gcr configure-docker --registries=us-central1-docker.pkg.dev

# echo "=== STEP 4: Pull Image ==="
# docker pull us-central1-docker.pkg.dev/datapipeline-infra/triallink-pipeline/triallink-pipeline:2026-02-16-042549

# echo "=== STEP 5: Upload log to GCS ==="
# gsutil cp $LOG_FILE gs://triallink-pipeline-data-datapipeline-infra/logs/startup-$(date -u +%Y%m%d-%H%M%S).log

# echo "=== STEP 6: Shutting down ==="
# shutdown -h now
# """,
        
        allow_stopping_for_update=True,
    )


    # Block all inbound SSH
    deny_ssh = gcp.compute.Firewall(
        "deny-ssh-pipeline",
        network="default",
        direction="INGRESS",
        priority=900,
        denies=[gcp.compute.FirewallDenyArgs(
            protocol="tcp",
            ports=["22"],
        )],
        source_ranges=["0.0.0.0/0"],
        target_tags=["pipeline-vm"],
    )

    # Allow only outbound HTTPS
    allow_outbound_https = gcp.compute.Firewall(
        "allow-outbound-https-pipeline",
        network="default",
        direction="EGRESS",
        priority=1000,
        allows=[gcp.compute.FirewallAllowArgs(
            protocol="tcp",
            ports=["443"],
        )],
        destination_ranges=["0.0.0.0/0"],
        target_tags=["pipeline-vm"],
    )

    # Cloud Function to start the VM
    function_archive = pulumi.AssetArchive({
        "main.py": pulumi.FileAsset("../../pipelines/cloud_functions/start_vm/main.py"),
        "requirements.txt": pulumi.FileAsset("../../pipelines/cloud_functions/start_vm/requirements.txt"),
    })

    function_bucket_object = gcp.storage.BucketObject(
        "start-vm-source",
        bucket=pipeline_bucket.name,
        source=function_archive,
        name="cloud-functions/start-vm-source.zip",
    )

    start_vm_function = gcp.cloudfunctionsv2.Function(
        "start-vm-function",
        name="triallink-start-pipeline-vm",
        location=region,
        build_config=gcp.cloudfunctionsv2.FunctionBuildConfigArgs(
            runtime="python311",
            entry_point="start_vm",
            source=gcp.cloudfunctionsv2.FunctionBuildConfigSourceArgs(
                storage_source=gcp.cloudfunctionsv2.FunctionBuildConfigSourceStorageSourceArgs(
                    bucket=pipeline_bucket.name,
                    object=function_bucket_object.name,
                ),
            ),
        ),
        service_config=gcp.cloudfunctionsv2.FunctionServiceConfigArgs(
            max_instance_count=1,
            available_memory="256M",
            timeout_seconds=60,
            service_account_email=pipeline_sa.email,
        ),
    )

    pulumi.export("start_vm_function_url", start_vm_function.url)

    pulumi.export("vm_name", pipeline_vm.name)
    pulumi.export("vm_zone", pipeline_vm.zone)

    pulumi.export("service_account_email", pipeline_sa.email)

    pulumi.export("registry_url", pulumi.Output.concat(
        region, "-docker.pkg.dev/", project, "/triallink-pipeline"
    ))

    pulumi.export("pipeline_bucket_name", pipeline_bucket.name)
    pulumi.export("pipeline_bucket_url", pipeline_bucket.url)