#!/bin/bash
# Cloud-init helper script used as Terraform `user_data` for the compute instance.
# The script assumes Terraform will template necessary OCI config or environment variables.

set -eux

# Basic tools
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg lsb-release unzip git apt-transport-https

# Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io
systemctl enable --now docker

# Python and OCI CLI
apt-get install -y python3 python3-pip
pip3 install --upgrade pip
pip3 install oci

# Optional: write OCI config if provided via environment variable (Terraform can template this)
if [ -n "${OCI_CONFIG_CONTENT:-}" ]; then
  mkdir -p /home/opc/.oci
  echo "$OCI_CONFIG_CONTENT" > /home/opc/.oci/config
  chown -R opc:opc /home/opc/.oci
fi

# Wait for docker to be ready
sleep 5

echo "Builder startup complete."

########################################
# Build & push image from repo tarball
########################################
if [ -n "${REPO_TARBALL:-}" ] && [ -n "${ARTIFACT_BUCKET:-}" ] && [ -n "${OCIR_USERNAME:-}" ] && [ -n "${OCIR_PASSWORD:-}" ]; then
  echo "Building image from repo tarball: ${REPO_TARBALL}"
  pip3 install oci
  mkdir -p /opt/repo /opt/artifacts
  # download tarball
  oci os object get --bucket-name "$ARTIFACT_BUCKET" --name "$REPO_TARBALL" --file /opt/repo/repo.tar.gz || true
  if [ -f /opt/repo/repo.tar.gz ]; then
    tar -xzf /opt/repo/repo.tar.gz -C /opt/repo || true
    cd /opt/repo
  elif [ -n "${GIT_REPO:-}" ]; then
    git clone "$GIT_REPO" /opt/repo || true
    cd /opt/repo
  else
    echo "No repo found to build (no repo tarball and no GIT_REPO)."
  fi

  # If Dockerfile exists, build and push
  if [ -f Dockerfile ]; then
    # Login to OCIR
    echo "$OCIR_PASSWORD" | docker login -u "$OCIR_USERNAME" --password-stdin ${OCIR_REGISTRY:-iad.ocir.io} || true
    # Build image tag; default to var supplied DOCKER_IMAGE env or fallback
    TAG=${DOCKER_IMAGE:-${OCIR_REGISTRY:-iad.ocir.io}/${OCIR_NAMESPACE:-idfx2prp9pwc}/emotion-spot:latest}
    docker build -t "$TAG" . || true
    docker push "$TAG" || true

    # Optionally download features and manifest to /opt/artifacts
    if [ -n "${ARTIFACT_BUCKET:-}" ]; then
      oci os object get --bucket-name "$ARTIFACT_BUCKET" --name "$FEATURES_ZIP" --file /opt/artifacts/features.zip || true
      oci os object get --bucket-name "$ARTIFACT_BUCKET" --name "$MANIFEST_ZIP" --file /opt/artifacts/manifest.zip || true
      if [ -f /opt/artifacts/features.zip ]; then
        unzip -o /opt/artifacts/features.zip -d /opt/artifacts || true
      fi
    fi

    # Run a smoke test if requested
    if [ "${RUN_SMOKE:-0}" = "1" ]; then
      docker run --rm -v /opt/artifacts:/data "$TAG" python train_multimodal_massive.py --manifest /data/manifests/multimodal_manifest.csv --epochs 1 || true
    fi
  else
    echo "No Dockerfile present in repo; skipping build."
  fi
fi
