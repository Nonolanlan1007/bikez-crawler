"""rustfs/S3 client. Sync (boto3) — the async crawler calls this via ``asyncio.to_thread``
so we only need to maintain one S3 implementation shared with the sync image pipeline."""

from __future__ import annotations

from collections.abc import Iterator

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


class S3Client:
    def __init__(self, *, endpoint: str, access_key: str, secret_key: str, region: str = "us-east-1") -> None:
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(s3={"addressing_style": "path"}),
        )

    def ensure_bucket(self, bucket: str) -> None:
        try:
            self._client.create_bucket(Bucket=bucket)
        except ClientError as err:
            code = err.response.get("Error", {}).get("Code")
            if code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                raise

    def upload(self, bucket: str, key: str, body: bytes, content_type: str | None = None) -> None:
        if content_type:
            self._client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
        else:
            self._client.put_object(Bucket=bucket, Key=key, Body=body)

    def download(self, bucket: str, key: str) -> bytes:
        response = self._client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

    def head(self, bucket: str, key: str) -> bool:
        try:
            self._client.head_object(Bucket=bucket, Key=key)
        except ClientError as err:
            if err.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return False
            raise
        return True

    def list_keys(self, bucket: str, prefix: str) -> Iterator[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]
