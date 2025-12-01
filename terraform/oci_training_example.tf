// Minimal Terraform example for creating an OCI instance that pulls the OCIR image
// This is a template: fill in provider credentials and variable values before use.

terraform {
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 4.0"
    }
  }
}

// (Provider and the first example instance removed in favor of a single
// instance block further below that includes templated startup user-data.)
/*
  Example Terraform for OCI: creates an Object Storage bucket and a Compute instance
  with a startup script to pull a Docker image and run the training command.

  Notes:
  - Fill in provider config and variables for your tenancy, region, and compartment.
  - This is a minimal example to bootstrap training. For production, harden networking,
    use IAM policies, and place instances in appropriate subnets.
*/

provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

// Provider variables are declared in `variables.tf`.

resource "oci_objectstorage_bucket" "features_bucket" {
  compartment_id = var.compartment_ocid
  name           = var.bucket_name
}

# Create a VCN and subnet if an existing subnet OCID is not provided.
resource "oci_core_vcn" "rcma_vcn" {
  compartment_id = var.compartment_ocid
  display_name   = "rcma-vcn"
  cidr_block     = var.vcn_cidr
}

resource "oci_core_subnet" "rcma_subnet" {
  compartment_id       = var.compartment_ocid
  vcn_id               = oci_core_vcn.rcma_vcn.id
  display_name         = "rcma-subnet"
  cidr_block           = var.subnet_cidr
  prohibit_public_ip_on_vnic = false
  dns_label            = "rcma"
}

// bucket_name variable is declared in `variables.tf`.

# Basic compute instance (replace shape with GPU shape if needed)
resource "oci_core_instance" "training_vm" {
  compartment_id = var.compartment_ocid
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  shape = var.instance_shape

  display_name = "rcma-training-builder"

  source_details {
    # Use an Oracle-provided Linux image or custom image OCID
    source_type = "image"
    image_id = var.image_id
  }

  metadata = {
    ssh_authorized_keys = file(var.ssh_pub_key_path)
    user_data = base64encode(templatefile("${path.module}/startup.sh", {
      GIT_REPO       = var.git_repo,
      REPO_TARBALL   = var.repo_tarball,
      ARTIFACT_BUCKET= var.bucket_name,
      OCIR_USERNAME  = var.ocir_user,
      OCIR_PASSWORD  = var.ocir_auth_token,
      OCIR_REGISTRY  = var.ocir_registry,
      DOCKER_IMAGE   = var.docker_image,
      RUN_SMOKE      = tostring(var.run_smoke),
      FEATURES_ZIP   = var.features_zip,
      MANIFEST_ZIP   = var.manifest_zip,
      OCIR_NAMESPACE = var.ocir_namespace
    }))
  }

  create_vnic_details {
    subnet_id = (var.subnet_id != "" && var.subnet_id != "<SUBNET_OCID>") ? var.subnet_id : oci_core_subnet.rcma_subnet.id
    assign_public_ip = true
  }
}
data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

output "bucket" {
  value = oci_objectstorage_bucket.features_bucket.name
}
