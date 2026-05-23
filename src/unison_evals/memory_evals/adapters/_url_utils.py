"""Shared URL utilities for adapter auth decisions."""

from __future__ import annotations

from urllib.parse import urlparse

LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"}


def is_localhost_url(url: str) -> bool:
    """True if the URL targets a local-only hostname (localhost, 127.0.0.1, etc.)."""
    if not url:
        return False
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = (parsed.hostname or "").lower()
    return host in LOCAL_HOSTNAMES
