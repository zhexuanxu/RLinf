# Copyright (c) 2025, RLinf contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained normalization stats + quantile (un)normalization.

Vendored re-implementation of the pieces of ``openpi.shared.normalize`` and the
quantile branches of ``openpi.transforms.Normalize`` / ``Unnormalize`` that the
BEHAVIOR pi05 eval path uses, so the package does not depend on the installed
``openpi`` distribution. The math is kept byte-identical to upstream (verified
by a cross-check test against the installed ``openpi``).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

# Matches openpi's `1e-6` denominator epsilon in the quantile (un)normalization.
_EPS = 1e-6


@dataclasses.dataclass
class NormStats:
    """Per-key normalization statistics (mean/std and 1st/99th quantiles)."""

    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


def load_norm_stats(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load ``norm_stats.json`` produced by openpi into ``{key: NormStats}``.

    The on-disk format is ``{"norm_stats": {key: {mean, std, q01, q99}}}``.
    """
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    data = json.loads(path.read_text())
    raw = data["norm_stats"] if "norm_stats" in data else data
    out: dict[str, NormStats] = {}
    for key, stats in raw.items():
        out[key] = NormStats(
            mean=np.asarray(stats["mean"]),
            std=np.asarray(stats["std"]),
            q01=np.asarray(stats["q01"]) if stats.get("q01") is not None else None,
            q99=np.asarray(stats["q99"]) if stats.get("q99") is not None else None,
        )
    return out


def _is_blank(value) -> bool:
    """True if ``value`` is ``None`` or an empty / whitespace-only string."""
    return value is None or (isinstance(value, str) and not value.strip())


def blank_asset_field(assets_dir, asset_id) -> str | None:
    """Return the name of the first blank asset field, or ``None`` if both are set.

    A field is "blank" when it is ``None`` or an empty / whitespace-only string.
    Shared by the eval model factory and the SFT data-loader builder so that every
    norm-stats entry point enforces the SAME non-empty YAML contract — a blank
    value is not a value and must never fall back to bare (non-task-0000) stats.
    Returns ``"assets_dir"`` or ``"asset_id"`` (assets_dir checked first) so the
    caller can raise an error naming exactly the missing field.
    """
    if _is_blank(assets_dir):
        return "assets_dir"
    if _is_blank(asset_id):
        return "asset_id"
    return None


def resolve_norm_stats_dir(
    assets_dir: pathlib.Path | str, asset_id: str | None
) -> pathlib.Path:
    """Return the directory holding ``norm_stats.json`` for ``(assets_dir, asset_id)``.

    Mirrors the BEHAVIOR asset layout: when ``asset_id`` is given, the stats live
    at EXACTLY ``{assets_dir}/{asset_id}/norm_stats.json`` (e.g.
    ``.../behavior-1k/2025-challenge-demos/norm_stats.json``). The bare
    ``{assets_dir}/norm_stats.json`` form is used ONLY when ``asset_id is None``
    (explicit direct-directory callers); an empty / whitespace-only ``asset_id``
    is rejected, since a blank YAML value must not silently resolve bare stats. A
    missing artifact raises rather than returning a different (non-task-0000)
    file. This is the shared resolution both the eval model factory and the SFT
    data loader use, so they always resolve the same canonical file (AC-8).
    """
    base = pathlib.Path(assets_dir).expanduser()
    if asset_id is None:
        directory = base
    elif _is_blank(asset_id):
        raise FileNotFoundError(
            f"BEHAVIOR norm stats require a non-empty asset_id (got {asset_id!r}); "
            "pass asset_id=None only for explicit direct-directory resolution."
        )
    else:
        directory = base / asset_id
    if (directory / "norm_stats.json").is_file():
        return directory
    raise FileNotFoundError(
        f"BEHAVIOR norm_stats.json not found at {directory / 'norm_stats.json'} "
        f"(assets_dir={str(base)!r}, asset_id={asset_id!r})."
    )


def normalize_quantile(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Map ``x`` to ``[-1, 1]`` using q01/q99 (openpi quantile normalize)."""
    if stats.q01 is None or stats.q99 is None:
        raise ValueError("Quantile normalization requires q01 and q99.")
    q01 = stats.q01[..., : x.shape[-1]]
    q99 = stats.q99[..., : x.shape[-1]]
    return (x - q01) / (q99 - q01 + _EPS) * 2.0 - 1.0


def unnormalize_quantile(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """Invert :func:`normalize_quantile` (openpi quantile unnormalize).

    If the stats cover fewer dims than ``x``, the trailing dims are passed
    through unchanged, matching openpi's behavior.
    """
    if stats.q01 is None or stats.q99 is None:
        raise ValueError("Quantile unnormalization requires q01 and q99.")
    q01, q99 = stats.q01, stats.q99
    dim = q01.shape[-1]
    if dim < x.shape[-1]:
        head = (x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + _EPS) + q01
        return np.concatenate([head, x[..., dim:]], axis=-1)
    return (x + 1.0) / 2.0 * (q99 - q01 + _EPS) + q01
