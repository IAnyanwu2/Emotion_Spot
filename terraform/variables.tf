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
