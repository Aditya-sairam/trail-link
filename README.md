# Trial-Link

MLOps platform for clinical trial matching.

## Setup

1. Clone the repository
2. Create and activate virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
```
3. Install dependencies:
```bash
   pip install -r requirements.txt
```

## Project Structure

- `infra/` - Infrastructure as code
- `sdk/` - Python SDK for ML operations
- `models/` - Model definitions and training scripts
- `pipelines/` - ML pipelines
- `tests/` - Test suite
- `configs/` - Configuration files

## How to setup and run pulumi

**Step 1**: Install pulumi
For MacOs :
```bash
   brew install pulumi/tap/pulumi
```

For Linux:
``` bash
curl -fsSL https://get.pulumi.com | sh
```

**Step 2:**  Login to pulumi (Please stick with the GCP bucket method)
```
gcloud storage buckets create gs://pulumi-state-YOUR_PROJECT_ID --location=US
pulumi login gs://pulumi-state-YOUR_PROJECT_ID
```
**step 3:**
```
cd infra/pulumi_stacks
```

**step 4:** Create stacks you want to push
```
pulumi preview #Previews the changes that will be pushed
pulumi up # Pushes the changes to infra
```

