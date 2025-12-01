Terraform example for OCI training runner

What this provides
- A minimal Terraform example (`oci_training_example.tf`) that creates a compute instance and injects `startup.sh` as `user_data`.

Notes / next steps
- This is a template: you must provide your provider auth (either via environment variables or the shared OCI config file) and fill the variable values.
- Set `OCI_CONFIG_CONTENT`, `OCIR_USERNAME`, `OCIR_PASSWORD`, `OCIR_IMAGE`, `ARTIFACT_BUCKET`, `FEATURES_ZIP`, and `MANIFEST_ZIP` via Terraform `user_data` templating or instance metadata.
- The `startup.sh` will attempt a single-epoch smoke run by default if it finds `OCIR_IMAGE` and artifact names.

Usage (high level)
1. Update `terraform/variables.tf` or pass variables at `terraform apply` time.
2. Run `terraform init` and `terraform apply -var 'compartment_id=...' -var 'subnet_id=...' -var 'image_ocid=...' -var 'ssh_public_key_path=~/.ssh/id_rsa.pub' -var 'availability_domain=...'`
