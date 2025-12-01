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

provider "oci" {
  # configure provider via environment or shared config file
  # region = var.region
}

// Example compute instance. Update shape / image and networking to match your tenancy.
resource "oci_core_instance" "training_vm" {
  compartment_id = var.compartment_id
  availability_domain = var.availability_domain
  shape = var.instance_shape

  display_name = "rcma-training-builder"

  source_details {
    # Use an Oracle-provided Linux image or custom image OCID
    source_type = "image"
    image_id = var.image_ocid
  }

  metadata = {
    ssh_authorized_keys = file(var.ssh_public_key_path)
    user_data = base64encode(file("${path.module}/startup.sh"))
  }

  create_vnic_details {
    subnet_id = var.subnet_id
    assign_public_ip = true
  }
}

output "instance_id" {
  value = oci_core_instance.training_vm.id
}
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

variable "tenancy_ocid" {}
variable "user_ocid" {}
variable "fingerprint" {}
variable "private_key_path" {}
variable "region" { default = "us-ashburn-1" }
variable "compartment_ocid" {}

resource "oci_objectstorage_bucket" "features_bucket" {
  compartment_id = var.compartment_ocid
  name           = var.bucket_name
}

variable "bucket_name" { default = "emotion-spot-features" }

# Basic compute instance (replace shape with GPU shape if needed)
resource "oci_core_instance" "train_instance" {
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  compartment_id      = var.compartment_ocid
  shape               = var.instance_shape

  create_vnic_details {
    subnet_id = var.subnet_id
  }

  metadata = {
    ssh_authorized_keys = file(var.ssh_pub_key_path)
    user_data = base64encode(<<-EOT
      #!/bin/bash
      set -e
      apt-get update
      apt-get install -y docker.io python3-pip unzip curl || true

      # Docker login to OCIR using auth token provided as a variable
      echo "Logging into OCIR registry ${var.ocir_registry} as ${var.ocir_user}"
      echo "${var.ocir_auth_token}" | docker login ${var.ocir_registry} -u "${var.ocir_user}" --password-stdin || true

      # Pull the training image
      docker pull ${var.docker_image} || true

      mkdir -p /home/opc/checkpoints

      # Run the training container. Environment vars point to Object Storage bucket and optional checkpoints.
      docker run --rm \
        -e OCIR_REGISTRY=${var.ocir_registry} \
        -e OCIR_USER=${var.ocir_user} \
        -e OCIR_AUTH_TOKEN='${var.ocir_auth_token}' \
        -e OCI_BUCKET=${oci_objectstorage_bucket.features_bucket.name} \
        -e FEATURES_PREFIX='${var.features_prefix}' \
        -e VISUAL_CHECKPOINT='${var.visual_checkpoint}' \
        -e AUDIO_CHECKPOINT='${var.audio_checkpoint}' \
        -v /home/opc/checkpoints:/app/checkpoints \
        ${var.docker_image} \
        python train_multimodal_rcma.py --epochs 1 --batch-size 4 --device cuda || true
    EOT
    )
  }

  source_details {
    source_type = "image"
    image_id    = var.image_id
  }

  display_name = "emotion-spot-train"
}

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

variable "instance_shape" { default = "VM.Standard.E3.Flex" }
variable "image_id" {}
variable "subnet_id" {}
variable "ssh_pub_key_path" {}
variable "docker_image" { default = "<region>.ocir.io/<tenancy-namespace>/emotion-spot:latest" }

variable "ocir_registry" { description = "OCIR registry host, e.g. iad.ocir.io" }
variable "ocir_user" { description = "OCIR username (usually <tenancy-namespace>/<user>)" }
variable "ocir_auth_token" { description = "OCIR auth token (sensitive)", sensitive = true }
variable "features_prefix" { description = "Object Storage prefix/folder where features & manifest are stored", default = "features/" }
variable "visual_checkpoint" { description = "Optional visual backbone checkpoint object name in bucket", default = "" }
variable "audio_checkpoint" { description = "Optional audio backbone checkpoint object name in bucket", default = "" }

output "bucket" {
  value = oci_objectstorage_bucket.features_bucket.name
}
