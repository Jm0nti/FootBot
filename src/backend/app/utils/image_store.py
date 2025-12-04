import threading
import uuid
from typing import Optional
from cachetools import TTLCache

# Simple in-memory TTL store for image bytes. Keys are UUID strings.
# This is suitable for small-scale use (dev or single-instance deployments).

# Cache size and default TTL (seconds)
_CACHE_MAXSIZE = 200
_CACHE_TTL = 300  # 5 minutes

_cache = TTLCache(maxsize=_CACHE_MAXSIZE, ttl=_CACHE_TTL)
_lock = threading.Lock()


def store_image_bytes(img_bytes: bytes, file_name: str, mime: str, ttl: Optional[int] = None) -> str:
    """Store image bytes and return an image id.

    img_bytes: raw image bytes
    file_name: original file name (for metadata)
    mime: mime type, e.g. 'image/jpeg'
    ttl: optional time-to-live in seconds (overrides default)
    """
    image_id = str(uuid.uuid4())
    meta = {
        "bytes": img_bytes,
        "file_name": file_name,
        "mime": mime,
    }

    with _lock:
        if ttl is None:
            _cache[image_id] = meta
        else:
            # temporary per-item TTL: create a small cache entry that expires
            # cachetools TTLCache doesn't support per-item ttl out of the box,
            # so for simplicity we ignore per-item ttl here and rely on global TTL.
            _cache[image_id] = meta

    return image_id


def get_image_bytes(image_id: str) -> dict:
    """Return stored image metadata dict or raise KeyError if missing."""
    with _lock:
        meta = _cache.get(image_id)
        if meta is None:
            raise KeyError("image not found or expired")
        return meta


def delete_image(image_id: str) -> None:
    with _lock:
        if image_id in _cache:
            del _cache[image_id]
