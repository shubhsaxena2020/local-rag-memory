"""Deterministic, order-independent dev / held-out splitting.

Assignment depends only on (seed, namespace, stratum, key) hashes and the number of items in
each stratum, never on input order, so re-running on the same data always reproduces the split.
Adding or removing items changes the per-stratum counts and may move other items; freeze the
`split.json` manifest written by `grounded_eval.py prepare` once a run starts.
"""

import hashlib
from typing import Any, Callable, Hashable, Iterable


def _digest(seed: int, namespace: str, stratum: Hashable, key: str) -> str:
    raw = f"{seed}\x1f{namespace}\x1f{stratum}\x1f{key}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def assign_splits(
    items: Iterable[Any],
    *,
    key: Callable[[Any], str],
    seed: int,
    namespace: str,
    dev_fraction: float = 0.5,
    stratum: Callable[[Any], Hashable] | None = None,
) -> dict[str, str]:
    """Return {key: "dev" | "heldout"}. Each stratum with >=2 items gets both splits."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    if isinstance(dev_fraction, bool) or not isinstance(dev_fraction, (int, float)) or not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be a number strictly between 0 and 1")
    groups: dict[Hashable, list[str]] = {}
    seen: set[str] = set()
    for item in items:
        k = key(item)
        if not isinstance(k, str) or not k.strip():
            raise ValueError("split keys must be non-empty strings")
        if k in seen:
            raise ValueError(f"duplicate split key: {k[:60]!r}")
        seen.add(k)
        groups.setdefault(stratum(item) if stratum else None, []).append(k)
    assignment: dict[str, str] = {}
    for group_name, keys in groups.items():
        ordered = sorted(keys, key=lambda k: _digest(seed, namespace, group_name, k))
        n = len(ordered)
        if n == 1:
            to_dev = int(_digest(seed, namespace, group_name, ordered[0])[:8], 16) % 2 == 0
            n_dev = 1 if to_dev else 0
        else:
            n_dev = min(n - 1, max(1, int(dev_fraction * n + 0.5)))
        for i, k in enumerate(ordered):
            assignment[k] = "dev" if i < n_dev else "heldout"
    return assignment


def split_dev_heldout(
    items: list[Any],
    *,
    key: Callable[[Any], str],
    seed: int,
    namespace: str,
    dev_fraction: float = 0.5,
    stratum: Callable[[Any], Hashable] | None = None,
) -> tuple[list[Any], list[Any]]:
    """Split items into (dev, heldout), preserving input order inside each list."""
    items = list(items)
    mapping = assign_splits(items, key=key, seed=seed, namespace=namespace, dev_fraction=dev_fraction, stratum=stratum)
    dev = [item for item in items if mapping[key(item)] == "dev"]
    heldout = [item for item in items if mapping[key(item)] == "heldout"]
    return dev, heldout
