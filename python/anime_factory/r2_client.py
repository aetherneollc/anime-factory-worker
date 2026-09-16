"""Optional S3-compatible put/list against R2. boto3 is optional."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def r2_env() -> dict[str, str]:
    return {
        "endpoint": os.environ.get("R2_ENDPOINT") or "",
        "bucket": os.environ.get("R2_BUCKET") or "",
        "access_key": os.environ.get("R2_ACCESS_KEY_ID") or "",
        "secret_key": os.environ.get("R2_SECRET_ACCESS_KEY") or "",
    }


def _client():
    cfg = r2_env()
    if not (cfg["endpoint"] and cfg["access_key"] and cfg["secret_key"]):
        return None, cfg
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        return None, cfg
    session = boto3.session.Session()
    client = session.client(
        "s3",
        endpoint_url=cfg["endpoint"],
        aws_access_key_id=cfg["access_key"],
        aws_secret_access_key=cfg["secret_key"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )
    return client, cfg


def put_file(key: str, path: Path, content_type: str = "application/octet-stream") -> dict[str, Any]:
    client, cfg = _client()
    if client is None:
        return {"ok": False, "skipped": True, "key": key, "reason": "boto3_or_creds_missing"}
    extra = {"ContentType": content_type}
    client.upload_file(str(path), cfg["bucket"], key, ExtraArgs=extra)
    return {"ok": True, "key": key, "bucket": cfg["bucket"]}


def upload_file(key: str, path: Path, content_type: str = "application/octet-stream") -> dict[str, Any]:
    return put_file(key, path, content_type)


def download_file(key: str, dest: Path) -> bool:
    client, cfg = _client()
    if client is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(cfg["bucket"], key, str(dest))
        return dest.is_file() and dest.stat().st_size > 0
    except Exception:  # noqa: BLE001
        return False


def list_prefix(prefix: str, max_keys: int = 2000) -> list[dict[str, Any]]:
    client, cfg = _client()
    if client is None:
        return []
    out: list[dict[str, Any]] = []
    token = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": cfg["bucket"], "Prefix": prefix, "MaxKeys": min(1000, max_keys)}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents") or []:
            out.append({"key": obj.get("Key"), "size": obj.get("Size")})
            if len(out) >= max_keys:
                return out
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return out


def list_all(prefix: str = "") -> list[dict[str, Any]]:
    """Paginate the whole prefix with no cap (for cache inventory)."""
    return list_prefix(prefix, max_keys=10_000_000)


def delete_keys(keys: list[str]) -> dict[str, Any]:
    client, cfg = _client()
    if client is None:
        return {"ok": False, "deleted": 0, "reason": "boto3_or_creds_missing"}
    deleted = 0
    batch: list[dict[str, str]] = []
    errors = 0
    for key in keys:
        batch.append({"Key": key})
        if len(batch) == 1000:
            resp = client.delete_objects(Bucket=cfg["bucket"], Delete={"Objects": batch, "Quiet": True})
            deleted += len(batch) - len(resp.get("Errors") or [])
            errors += len(resp.get("Errors") or [])
            batch = []
    if batch:
        resp = client.delete_objects(Bucket=cfg["bucket"], Delete={"Objects": batch, "Quiet": True})
        deleted += len(batch) - len(resp.get("Errors") or [])
        errors += len(resp.get("Errors") or [])
    return {"ok": errors == 0, "deleted": deleted, "errors": errors, "bucket": cfg["bucket"]}


def download_prefix(prefix: str, dest_root: Path, strip_prefix: str | None = None) -> list[str]:
    """Download R2 keys under prefix into dest_root, preserving relative paths."""
    keys = list_prefix(prefix)
    cut = strip_prefix if strip_prefix is not None else prefix
    written = []
    for item in keys:
        key = item.get("key") or ""
        if not key or key.endswith("/"):
            continue
        rel = key[len(cut) :] if key.startswith(cut) else key
        dest = dest_root / rel
        if download_file(key, dest):
            written.append(rel)
    return written


def upload_tree(story_id: str, story_root: Path, rel_dir: str = "assets") -> list[dict[str, Any]]:
    from anime_factory.r2_paths import join_story

    root = story_root / rel_dir
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(story_root).as_posix()
        key = join_story(story_id, rel)
        ctype = "image/png" if path.suffix.lower() == ".png" else "application/json" if path.suffix.lower() == ".json" else "application/octet-stream"
        out.append(put_file(key, path, ctype))
    return out
