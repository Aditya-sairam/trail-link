from googleapiclient import discovery


def start_vm(request):
    """Starts the pipeline VM."""
    project = "datapipeline-infra"
    zone = "us-central1-a"
    instance = "triallink-pipeline-vm"

    compute = discovery.build("compute", "v1")
    compute.instances().start(
        project=project, zone=zone, instance=instance
    ).execute()

    return f"Started {instance}", 200
