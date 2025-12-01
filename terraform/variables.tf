variable "compartment_id" {
  description = "Compartment OCID where the instance will be created"
  type        = string
}

variable "availability_domain" {
  description = "Availability domain"
  type        = string
}

variable "subnet_id" {
  description = "Subnet OCID for the instance VNIC"
  type        = string
}

variable "ssh_public_key_path" {
  description = "Path to the SSH public key to inject for 'opc' or user"
  type        = string
}

variable "image_ocid" {
  description = "Image OCID to use for the instance (Ubuntu/CentOS/OracleLinux)"
  type        = string
}

variable "instance_shape" {
  description = "Instance shape to use (e.g., VM.Standard.E4.Flex)"
  type        = string
  default     = "VM.Standard.E4.Flex"
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

variable "features_zip" { description = "Features archive name" default = "features.zip" }
variable "manifest_zip" { description = "Manifest archive name" default = "manifest.zip" }
variable "ocir_namespace" { description = "OCIR tenancy namespace" default = "idfx2prp9pwc" }
