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
  namespace      = var.ocir_namespace
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
    source_id = var.image_id
  }

  metadata = {
    ssh_authorized_keys = file(var.ssh_pub_key_path)
    user_data = base64encode(
      replace(
        replace(
          replace(
            replace(
              replace(
                replace(
                  replace(
                    replace(
                      replace(
                        replace(
                          file("${path.module}/startup.sh"),
                          "__GIT_REPO__", var.git_repo
                        ),
                        "__REPO_TARBALL__", var.repo_tarball
                      ),
                      "__ARTIFACT_BUCKET__", var.bucket_name
                    ),
                    "__OCIR_USERNAME__", var.ocir_user
                  ),
                  "__OCIR_PASSWORD__", var.ocir_auth_token
                ),
                "__OCIR_REGISTRY__", var.ocir_registry
              ),
              "__DOCKER_IMAGE__", var.docker_image
            ),
            "__RUN_SMOKE__", tostring(var.run_smoke)
          ),
          "__FEATURES_ZIP__", var.features_zip
        ),
        "__MANIFEST_ZIP__", var.manifest_zip
      )
    )
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
