"""
Ollama loopback HTTP client for the embedding benchmark harness.

Enforces 100% local loopback communication, performs /api/embed calls,
validates vector dimensions and finite numbers, inspects resident/available models,
and provides safe model unloading without deleting models.
"""

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
ALLOWED_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
MAX_INPUT_TEXT_LENGTH = 32768


def validate_loopback_url(url: str) -> None:
    """Validate that the given URL strictly targets local loopback over HTTP."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Ollama base URL must be a non-empty string")
    parsed = urllib.parse.urlsplit(url.strip())
    if parsed.scheme.lower() != "http":
        raise ValueError(f"Ollama base URL must use HTTP scheme, got: '{parsed.scheme}'")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Ollama base URL must not contain user information")
    if parsed.query or parsed.fragment:
        raise ValueError("Ollama base URL must not contain a query or fragment")
    hostname = (parsed.hostname or "").lower()
    if hostname not in ALLOWED_LOOPBACK_HOSTS:
        raise ValueError(
            f"Ollama base URL must target a loopback host {ALLOWED_LOOPBACK_HOSTS}, got: '{hostname}'"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Ollama base URL has an invalid port") from exc
    if port == 0:
        raise ValueError("Ollama base URL port must be between 1 and 65535")


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_embedding(vec: List[float], expected_dim: Optional[int] = None) -> int:
    """Validate that an embedding vector is a non-empty list of finite numbers.

    Checks:
    - Must be a list or tuple of numbers (int or float).
    - Cannot be empty.
    - All elements must be finite numbers (no NaN, no +Inf, no -Inf).
    - If expected_dim is specified, length must match expected_dim.

    Returns the dimension (len(vec)).
    Raises ValueError on any violation.
    """
    if not isinstance(vec, (list, tuple)):
        raise ValueError(f"Embedding vector must be a list or tuple, got: {type(vec).__name__}")
    dim = len(vec)
    if dim == 0:
        raise ValueError("Embedding vector cannot be empty")
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(f"Embedding dimension mismatch: expected {expected_dim}, got {dim}")
    for idx, val in enumerate(vec):
        if not isinstance(val, (int, float)):
            raise ValueError(f"Embedding element at index {idx} is not numeric: {type(val).__name__}")
        if not math.isfinite(val):
            raise ValueError(f"Embedding element at index {idx} is not finite: {val}")
    return dim


class OllamaEmbeddingClient:
    """Concise loopback client for Ollama embedding and model lifecycle inspection."""

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 60.0):
        validate_loopback_url(base_url)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        method: str = "GET",
    ) -> Dict[str, Any]:
        """Perform a JSON HTTP request to Ollama loopback."""
        url = f"{self.base_url}{endpoint}"
        validate_loopback_url(url)

        headers = {"Accept": "application/json"}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirectHandler)
            with opener.open(req, timeout=self.timeout) as resp:
                raw_body = resp.read().decode("utf-8")
                if not raw_body.strip():
                    return {}
                return json.loads(raw_body)
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(f"Ollama HTTP error {e.code} on {endpoint}: {err_body or e.reason}")
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to local Ollama at {self.base_url}: {e.reason}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON returned from Ollama {endpoint}: {e}")

    def check_health(self) -> Dict[str, Any]:
        """Check loopback Ollama availability and report status."""
        try:
            data = self._request("/api/version", method="GET")
            return {"connected": True, "version": data.get("version", "unknown"), "url": self.base_url}
        except Exception as e:
            try:
                data = self._request("/api/tags", method="GET")
                return {"connected": True, "version": "unknown", "url": self.base_url}
            except Exception as e2:
                return {"connected": False, "error": str(e2), "url": self.base_url}

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Retrieve list of locally pulled models from /api/tags."""
        data = self._request("/api/tags", method="GET")
        return data.get("models", [])

    def get_available_model_names(self) -> Set[str]:
        """Retrieve normalized set of pulled model names."""
        models = self.get_available_models()
        names = set()
        for item in models:
            name = item.get("name", "")
            if name:
                names.add(name)
                # If name ends with :latest, also register the bare name
                if name.endswith(":latest"):
                    names.add(name[:-7])
                elif ":" not in name:
                    names.add(f"{name}:latest")
        return names

    def get_running_models(self) -> Set[str]:
        """Retrieve set of models currently loaded in RAM/VRAM via /api/ps."""
        try:
            data = self._request("/api/ps", method="GET")
            running = set()
            for item in data.get("models", []):
                name = item.get("name") or item.get("model", "")
                if name:
                    running.add(name)
                    if name.endswith(":latest"):
                        running.add(name[:-7])
                    elif ":" not in name:
                        running.add(f"{name}:latest")
            return running
        except Exception:
            # If /api/ps is unsupported on older versions, return empty set safely
            return set()

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Fetch model metadata (parameter size, quantization, file size) via /api/show."""
        try:
            return self._request("/api/show", payload={"name": model_name}, method="POST")
        except Exception:
            return {}

    def embed(
        self,
        model: str,
        input_data: str | List[str],
        truncate: bool = True,
        keep_alive: str | int | None = None,
    ) -> Tuple[List[List[float]], float]:
        """Call Ollama /api/embed endpoint to embed text or list of texts.

        Returns (embeddings_list, elapsed_seconds).
        Validates all returned vectors for dimensions and finite values.
        """
        if isinstance(input_data, str):
            inputs = [input_data]
            single_item = True
        elif isinstance(input_data, (list, tuple)):
            inputs = list(input_data)
            single_item = False
        else:
            raise ValueError(f"Input must be string or list of strings, got: {type(input_data).__name__}")

        if not inputs:
            return [], 0.0

        # Validate text length limits
        for idx, text in enumerate(inputs):
            if not isinstance(text, str):
                raise ValueError(f"Input item at index {idx} must be a string")
            if len(text) > MAX_INPUT_TEXT_LENGTH:
                # Bound input text to avoid crashing backend
                inputs[idx] = text[:MAX_INPUT_TEXT_LENGTH]

        payload: Dict[str, Any] = {
            "model": model,
            "input": inputs[0] if single_item else inputs,
            "truncate": truncate,
        }
        if keep_alive is not None:
            if isinstance(keep_alive, bool) or not isinstance(keep_alive, (str, int)):
                raise ValueError("keep_alive must be a string or integer")
            payload["keep_alive"] = keep_alive

        t0 = time.perf_counter()
        resp = self._request("/api/embed", payload=payload, method="POST")
        elapsed = time.perf_counter() - t0

        embeddings: List[List[float]] = []
        if "embeddings" in resp:
            raw_embeddings = resp["embeddings"]
            if isinstance(raw_embeddings, list):
                if raw_embeddings and isinstance(raw_embeddings[0], (int, float)):
                    # Single 1D vector returned
                    embeddings = [raw_embeddings]
                else:
                    embeddings = raw_embeddings
        elif "embedding" in resp:
            # Fallback for /api/embeddings compatibility if /api/embed was aliased
            embeddings = [resp["embedding"]]
        else:
            raise ValueError(f"No embeddings found in Ollama response for model '{model}'")

        if len(embeddings) != len(inputs):
            raise ValueError(
                f"Expected {len(inputs)} embedding vectors from model '{model}', but received {len(embeddings)}"
            )

        # Validate every returned vector
        expected_dim = None
        for i, vec in enumerate(embeddings):
            dim = validate_embedding(vec, expected_dim=expected_dim)
            if expected_dim is None:
                expected_dim = dim

        return embeddings, elapsed

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory using Ollama's keep_alive: 0 mechanism.

        Does NOT delete the model from disk.
        Returns True if successful, False otherwise.
        """
        try:
            # Ollama unloads a model when a generation request is sent with keep_alive=0
            payload = {"model": model_name, "prompt": "", "keep_alive": 0}
            self._request("/api/generate", payload=payload, method="POST")
            return True
        except Exception:
            try:
                # Alternative chat endpoint with keep_alive: 0
                payload = {"model": model_name, "messages": [], "keep_alive": 0}
                self._request("/api/chat", payload=payload, method="POST")
                return True
            except Exception:
                return False
