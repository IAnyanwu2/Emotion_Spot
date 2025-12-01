#!/usr/bin/env python3
"""List multipart uploads and parts in an OCI bucket.

Usage:
  python scripts/list_multipart_parts.py --bucket <bucket> [--upload-id <upload-id>] [--config <path>] [--profile DEFAULT]

If --upload-id is omitted the script lists active multipart uploads in the bucket.
If --upload-id is provided the script lists the parts that the server has received for that upload
and prints the count and an estimated total uploaded bytes.

Requires the OCI Python SDK (`pip install oci`).
"""
import argparse
import sys
import oci


def parse_args():
    p = argparse.ArgumentParser(description="List multipart uploads/parts in an OCI bucket")
    p.add_argument('--bucket', '-b', required=True, help='Bucket name')
    p.add_argument('--upload-id', '-u', help='Specific upload ID to inspect')
    p.add_argument('--config', default=None, help='Path to OCI config (~/.oci/config)')
    p.add_argument('--profile', default='DEFAULT', help='OCI config profile name')
    return p.parse_args()


def main():
    args = parse_args()

    try:
        if args.config:
            config = oci.config.from_file(args.config, args.profile)
        else:
            config = oci.config.from_file(profile_name=args.profile)
    except Exception as e:
        print('ERROR: failed to load OCI config:', e)
        sys.exit(2)

    client = oci.object_storage.ObjectStorageClient(config)

    try:
        namespace = client.get_namespace().data
    except Exception as e:
        print('ERROR: failed to get namespace:', e)
        sys.exit(2)

    bucket = args.bucket

    if not args.upload_id:
        # list active multipart uploads
        print(f'Listing active multipart uploads in bucket "{bucket}" (namespace={namespace})')
        page = None
        found = False
        while True:
            resp = client.list_multipart_uploads(namespace, bucket, page=page)
            uploads = resp.data
            if not uploads:
                break
            for u in uploads:
                found = True
                # try to extract common attributes safely
                upload_id = getattr(u, 'upload_id', None) or getattr(u, 'uploadId', None)
                object_name = getattr(u, 'object', None) or getattr(u, 'object_name', None) or getattr(u, 'objectName', None)
                time_created = getattr(u, 'time_created', None) or getattr(u, 'timeCreated', None)
                print(f'- UploadId: {upload_id}   Object: {object_name}   TimeCreated: {time_created}')
            if hasattr(resp, 'next_page') and resp.next_page:
                page = resp.next_page
            else:
                break
        if not found:
            print('No active multipart uploads found in the bucket.')
        sys.exit(0)

    # inspect parts for a given upload id
    upload_id = args.upload_id
    # Some OCI SDK versions require the object name when listing multipart upload parts.
    # If the user didn't pass an object name, discover it by listing active multipart uploads and
    # matching the upload id.
    object_name = None
    try:
        # try to find object name for the upload id
        page = None
        while True:
            resp = client.list_multipart_uploads(namespace, bucket, page=page)
            uploads = resp.data
            if not uploads:
                break
            for u in uploads:
                uid = getattr(u, 'upload_id', None) or getattr(u, 'uploadId', None)
                obj = getattr(u, 'object', None) or getattr(u, 'object_name', None) or getattr(u, 'objectName', None)
                if uid == upload_id:
                    object_name = obj
                    break
            if object_name:
                break
            if hasattr(resp, 'next_page') and resp.next_page:
                page = resp.next_page
            else:
                break
    except Exception:
        object_name = None

    if object_name is None:
        print(f'Could not discover object name for upload id={upload_id}.')
        print('You can re-run without --upload-id to list active uploads and their object names:')
        print(f'  python scripts/list_multipart_parts.py --bucket {bucket}')
        sys.exit(2)

    print(f'Listing parts for upload id={upload_id} (object="{object_name}") in bucket "{bucket}" (namespace={namespace})')
    page = None
    total_parts = 0
    total_bytes = 0
    while True:
        try:
            # most SDK versions require object_name as the third positional parameter
            resp = client.list_multipart_upload_parts(namespace, bucket, object_name, upload_id, page=page)
        except TypeError:
            # try the keyword-argument form some SDK builds expect
            resp = client.list_multipart_upload_parts(namespace_name=namespace, bucket_name=bucket, object_name=object_name, upload_id=upload_id, page=page)
        parts = resp.data
        if not parts:
            break
        for p in parts:
            total_parts += 1
            # try several common attribute names for part size
            size = getattr(p, 'size', None) or getattr(p, 'part_size', None) or getattr(p, 'content_length', None)
            if size is None:
                # best-effort: try to read dict-like
                try:
                    size = p['size']
                except Exception:
                    size = 0
            try:
                total_bytes += int(size or 0)
            except Exception:
                pass
            part_number = getattr(p, 'part_number', None) or getattr(p, 'partNumber', None)
            print(f'  Part {part_number}  size={size}')
        if hasattr(resp, 'next_page') and resp.next_page:
            page = resp.next_page
        else:
            break

    print(f'Parts found: {total_parts}')
    print(f'Estimated total uploaded bytes (sum of part sizes): {total_bytes} ({total_bytes/1024/1024:.2f} MiB)')


if __name__ == '__main__':
    main()
