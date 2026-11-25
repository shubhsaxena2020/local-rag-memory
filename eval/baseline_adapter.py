"""Read-only adapter for a local loopback baseline service.

No environment/file reads and no network I/O at import. URL, tenant ID and key
are passed explicitly; the key is only ever sent in the Authorization header of
a request to a validated loopback address and never appears in errors.
"""

import ipaddress
import json
import re
import urllib.error
import urllib.parse
import urllib.request

MAX_RESPONSE_BYTES = 1_048_576
DEFAULT_TIMEOUT = 5.0
_TENANT_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_KEY_RE = re.compile(r"[\x21-\x7e]{1,512}")  # printable ASCII, no whitespace


class BaselineError(RuntimeError):
    """Failure with a fixed, credential-free message."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # never forward the Authorization header elsewhere


# Empty ProxyHandler: ignore HTTP(S)_PROXY env so the key never leaves loopback.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


def validate_base_url(url: str) -> str:
    """Return normalized http://host:port for a loopback IP literal; else ValueError."""
    if not isinstance(url, str):
        raise ValueError("base URL must be a string")
    try:
        parts = urllib.parse.urlsplit(url)
        host, port = parts.hostname, parts.port
    except ValueError:
        raise ValueError("base URL is malformed") from None
    if parts.scheme != "http":
        raise ValueError("base URL must use http")
    if parts.username is not None or parts.password is not None:
        raise ValueError("base URL must not contain credentials")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ValueError("base URL must be host:port only")
    if port is None or not 1024 <= port <= 65535:
        raise ValueError("base URL must include an explicit port from 1024 to 65535")
    try:
        if not host or not ipaddress.ip_address(host).is_loopback:
            raise ValueError
    except ValueError:
        raise ValueError("base URL host must be a loopback IP literal (127.x.x.x or ::1)") from None
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"


def memory_recall(question: str, top_k: int = 5, *, base_url: str, tenant_id: str,
                  key: str, timeout: float = DEFAULT_TIMEOUT) -> list[dict]:
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        raise ValueError("question must be non-empty and at most 4000 characters")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be an integer from 1 to 100")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0.1 <= timeout <= 30:
        raise ValueError("timeout must be a number from 0.1 to 30 seconds")
    if not isinstance(tenant_id, str) or not _TENANT_RE.fullmatch(tenant_id):
        raise ValueError("tenant_id must be 1-64 characters of [A-Za-z0-9_-]")
    if not isinstance(key, str) or not _KEY_RE.fullmatch(key):
        raise ValueError("key must be 1-512 printable non-space ASCII characters")
    root = validate_base_url(base_url)
    request = urllib.request.Request(
        f"{root}/api/v1/{tenant_id}/query",
        data=json.dumps({"question": question, "top_k": top_k}).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        exc.close()
        raise BaselineError(f"Baseline recall request failed (HTTP {int(exc.code)})") from None
    except (urllib.error.URLError, OSError):  # includes timeouts
        raise BaselineError("Baseline recall request failed (connection error or timeout)") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise BaselineError("Baseline response too large")
    try:
        body = json.loads(raw)
    except ValueError:
        raise BaselineError("Baseline returned invalid JSON") from None
    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list) or not all(isinstance(r, dict) for r in results):
        raise BaselineError("Baseline returned an invalid results payload")
    return results[:top_k]
