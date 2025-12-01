// Variables aligned with terraform.tfvars used in this workspace.

variable "tenancy_ocid" {
  description = "Tenancy OCID"
  type        = string
}

variable "user_ocid" {
  description = "User OCID used for API key"
  type        = string
}

variable "fingerprint" {
  description = "Fingerprint for the API key"
  type        = string
}

variable "private_key_path" {
  description = "Path to the private key file used by the provider"
  type        = string
}

variable "region" {
  description = "OCI region"
  type        = string
  default     = "us-ashburn-1"
}

variable "compartment_ocid" {
  description = "Compartment OCID where resources will be created"
  type        = string
}

variable "bucket_name" {
  description = "Object Storage bucket name for artifacts"
  type        = string
  default     = "emotion-spot-features"
}

variable "image_id" {
  description = "Image OCID to use for the instance"
  type        = string
}

variable "subnet_id" {
  description = "Optional existing subnet OCID; leave as \"<SUBNET_OCID>\" or empty to create one"
  type        = string
  default     = "<SUBNET_OCID>"
}

variable "ssh_pub_key_path" {
  description = "Path to the SSH public key to inject into the instance (public key file)"
  type        = string
}

variable "instance_shape" {
  description = "Instance shape to use (e.g., VM.Standard.E3.Flex)"
  type        = string
  default     = "VM.Standard.E3.Flex"
}

variable "vcn_cidr" {
  description = "CIDR for the VCN created when no subnet is supplied"
  type        = string
  default     = "10.0.0.0/16"
}

variable "subnet_cidr" {
  description = "CIDR for the subnet created when no subnet is supplied"
  type        = string
  default     = "10.0.1.0/24"
}

variable "docker_image" {
  description = "Default docker image tag to push/pull"
  type        = string
  default     = "<region>.ocir.io/<tenancy-namespace>/emotion-spot:latest"
}

variable "ocir_registry" {
  description = "OCIR registry host, e.g. iad.ocir.io"
  type        = string
}

variable "ocir_user" {
  description = "OCIR username (usually <tenancy-namespace>/<user>)"
  type        = string
}

variable "ocir_auth_token" {
  description = "OCIR auth token (sensitive)"
  type        = string
  sensitive   = true
}

variable "ocir_namespace" {
  description = "OCIR tenancy namespace"
  type        = string
  default     = "idfx2prp9pwc"
}

variable "git_repo" {
  description = "Optional Git clone URL for the builder to pull the repo"
  type        = string
  default     = ""
}

variable "repo_tarball" {
  description = "Optional repo tarball object name in the bucket (repo.tar.gz)"
  type        = string
  default     = ""
}

variable "run_smoke" {
  description = "Whether to run the one-epoch smoke test (1=yes, 0=no)"
  type        = number
  default     = 0
}

variable "features_zip" {
  description = "Features archive name"
  type        = string
  default     = "features.zip"
}

variable "manifest_zip" {
  description = "Manifest archive name"
  type        = string
  default     = "manifest.zip"
}

variable "features_prefix" {
  description = "Object Storage prefix/folder where features & manifest are stored"
  type        = string
  default     = "features/"
}

variable "visual_checkpoint" {
  description = "Optional visual backbone checkpoint object name in bucket"
  type        = string
  default     = ""
}

variable "audio_checkpoint" {
  description = "Optional audio backbone checkpoint object name in bucket"
  type        = string
  default     = ""
}
