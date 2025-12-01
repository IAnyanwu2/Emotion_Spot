#!/usr/bin/env python3
"""Robust upload helper using OCI Python SDK UploadManager.

Usage:
  python scripts/upload_with_manager.py --file <path> --bucket <bucket-name> [--name <object-name>] [--part-size-mb 32] [--concurrency 4]

This script picks up OCI config from the default location (`~/.oci/config`).
It will print progress and retry transient failures.
"""
import argparse
import os
import sys
import math
import oci

# UploadManager import can live in different places depending on OCI SDK version.
try:
    from oci.object_storage.transfer.upload_manager import UploadManager
except Exception:
    try:
        from oci.object_storage.transfer import UploadManager
    except Exception:
        UploadManager = None


def parse_args():
    p = argparse.ArgumentParser(description="Upload large files to OCI Object Storage using UploadManager")
    p.add_argument("--file", "-f", required=True, help="Path to local file to upload")
    p.add_argument("--bucket", "-b", required=True, help="Bucket name")
    p.add_argument("--name", "-n", help="Object name in bucket (defaults to basename of file)")
    p.add_argument("--part-size-mb", type=int, default=32, help="Part size in MiB (default: 32)")
    p.add_argument("--concurrency", type=int, default=4, help="Number of concurrent upload threads (default: 4)")
    p.add_argument("--config", default=None, help="Path to OCI config file (defaults to ~/.oci/config)")
    p.add_argument("--profile", default="DEFAULT", help="OCI config profile name (default: DEFAULT)")
    return p.parse_args()


def human_size(n):
    for unit in ['B','KiB','MiB','GiB','TiB']:
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PiB"


def main():
    args = parse_args()
    file_path = os.path.expanduser(args.file)
    if not os.path.isfile(file_path):
        print(f"ERROR: file not found: {file_path}")
        sys.exit(2)

    object_name = args.name if args.name else os.path.basename(file_path)
    part_size_bytes = args.part_size_mb * 1024 * 1024

    print(f"Using OCI config: {args.config or '~/.oci/config'} profile: {args.profile}")
    print(f"Uploading {file_path} ({human_size(os.path.getsize(file_path))}) to bucket '{args.bucket}' as '{object_name}'")
    print(f"Part size: {args.part_size_mb} MiB, concurrency: {args.concurrency}")

    # Load OCI config and create client
    try:
        if args.config:
            config = oci.config.from_file(args.config, args.profile)
        else:
            config = oci.config.from_file(profile_name=args.profile)
    except Exception as e:
        print("ERROR: failed to load OCI config:", e)
        sys.exit(2)

    client = oci.object_storage.ObjectStorageClient(config)

    # get namespace
    try:
        ns = client.get_namespace().data
    except Exception as e:
        print("ERROR: failed to get namespace:", e)
        sys.exit(2)

    if UploadManager is None:
        print("ERROR: your installed 'oci' Python SDK does not provide UploadManager.")
        print("Please upgrade the SDK and retry, for example:")
        print()
        print(r"  & 'C:\Path\To\python.exe' -m pip install --upgrade oci")
        print()
        print("Or use your environment's python executable. After upgrading, rerun this script.")
        sys.exit(3)

    upload_manager = UploadManager(client, allow_resume=False)

    # Determine number of parts
    file_size = os.path.getsize(file_path)
    parts = math.ceil(file_size / part_size_bytes)
    if parts < 1:
        parts = 1

    print(f"Namespace: {ns} -- estimated parts: {parts}")

    try:
        # UploadManager.upload_file signature: upload_file(namespace, bucket_name, object_name, file_path, **kwargs)
        upload_manager.upload_file(ns, args.bucket, object_name, file_path, part_size=part_size_bytes, max_concurrency=args.concurrency)
        print("Upload finished successfully.")
    except Exception as e:
        print("ERROR: upload failed:", e)
        sys.exit(1)


if __name__ == '__main__':
    main()
