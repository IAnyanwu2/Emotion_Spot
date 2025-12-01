#!/usr/bin/env bash
# Upload features and manifest to OCI Object Storage using OCI CLI
# Assumes OCI CLI is configured and user has a bucket ready.

set -e

BUCKET="${OCI_BUCKET:-your_bucket_name}"
NAMESPACE="${OCI_NAMESPACE:-your_namespace}"
PROFILE="${OCI_PROFILE:-DEFAULT}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Uploading features and manifests from $ROOT_DIR to oci://$BUCKET@${NAMESPACE}"

oci os object bulk-upload --bucket-name $BUCKET --src-dir "$ROOT_DIR/features" --profile $PROFILE || true
oci os object put --bucket-name $BUCKET --file "$ROOT_DIR/manifests/multimodal_manifest.csv" --name manifests/multimodal_manifest.csv --profile $PROFILE || true

echo "Upload complete."
