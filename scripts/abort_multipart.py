#!/usr/bin/env python3
"""Abort an OCI multipart upload for an object using the Python SDK.

Usage:
  python .\scripts\abort_multipart.py --bucket <bucket> --name <object-name> --upload-id <upload-id>

Optional:
  --config <path to ~/.oci/config>
  --profile <profile name>

"""
import argparse
import sys
import oci


def parse_args():
    p = argparse.ArgumentParser(description="Abort OCI multipart upload")
    p.add_argument("--bucket", "-b", required=True, help="Bucket name")
    p.add_argument("--name", "-n", required=True, help="Object name")
    p.add_argument("--upload-id", "-u", required=True, help="Upload ID to abort")
    p.add_argument("--config", default=None, help="OCI config file path (defaults to ~/.oci/config)")
    p.add_argument("--profile", default="DEFAULT", help="OCI config profile name")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        if args.config:
            config = oci.config.from_file(args.config, args.profile)
        else:
            config = oci.config.from_file(profile_name=args.profile)
    except Exception as e:
        print("ERROR: failed to load OCI config:", e)
        sys.exit(2)

    client = oci.object_storage.ObjectStorageClient(config)

    try:
        namespace = client.get_namespace().data
    except Exception as e:
        print("ERROR: failed to get namespace:", e)
        sys.exit(2)

    try:
        print(f"Aborting upload id={args.upload_id} for object '{args.name}' in bucket '{args.bucket}' (namespace={namespace})")
        client.abort_multipart_upload(namespace_name=namespace, bucket_name=args.bucket, object_name=args.name, upload_id=args.upload_id)
        print("Abort request sent successfully.")
    except oci.exceptions.ServiceError as se:
        print("Service error:", se)
        sys.exit(3)
    except Exception as e:
        print("ERROR: aborted with exception:", e)
        sys.exit(1)


if __name__ == '__main__':
    main()
