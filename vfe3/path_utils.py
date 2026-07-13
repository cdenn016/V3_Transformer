"""Deterministic filesystem-safe names for user- and experiment-supplied labels."""

import hashlib
import re


def filesystem_slug(
    value: str,

    *,
    fallback: str = "artifact",
) -> str:
    r"""Return one safe path component plus a stable collision-disambiguating hash.

    The readable prefix uses one allowlist across every artifact producer. The suffix binds the
    original value, so distinct inputs remain distinct even when their readable prefixes collide.
    """
    if not isinstance(value, str):
        raise TypeError(f"filesystem_slug value must be str, got {type(value).__name__}")
    if not isinstance(fallback, str) or not fallback:
        raise ValueError("filesystem_slug fallback must be a nonempty string")
    safe_fallback = re.sub(r"[^A-Za-z0-9._-]", "_", fallback).strip("._")
    if not safe_fallback:
        raise ValueError("filesystem_slug fallback must contain a filesystem-safe character")
    prefix = re.sub(r"[^A-Za-z0-9._-]", "_", value).strip("._") or safe_fallback
    prefix = prefix[:120]
    suffix = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}__{suffix}"
