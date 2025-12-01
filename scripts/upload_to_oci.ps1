Param(
    [string]$Bucket = $env:OCI_BUCKET,
    [string]$Namespace = $env:OCI_NAMESPACE,
    [string]$Profile = $env:OCI_PROFILE
)

if (-not $Bucket) { Write-Error "Set OCI_BUCKET env var or pass -Bucket"; exit 1 }
if (-not $Namespace) { Write-Error "Set OCI_NAMESPACE env var or pass -Namespace"; exit 1 }

$RootDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "Uploading features and manifest from $RootDir to oci://$Bucket@$Namespace"

# Bulk upload features folder
oci os object bulk-upload --bucket-name $Bucket --src-dir "$RootDir/features" --profile ${Profile} || Write-Host "Bulk upload failed"
oci os object put --bucket-name $Bucket --file "$RootDir/manifests/multimodal_manifest.csv" --name manifests/multimodal_manifest.csv --profile ${Profile} || Write-Host "Manifest upload failed"

Write-Host "Upload complete."
