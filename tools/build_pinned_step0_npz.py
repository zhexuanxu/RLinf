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

"""Convert the step-0 2x2 reference dump into the pinned-loader ``.npz`` format.

This is the reproducible bridge from the committed reference dumper
(``tests/unit_tests/_ref_step0_2x2_dump.py``) to the reproducibility-only pinned
SFT loader (``rlinf/data/datasets/behavior/behavior_pinned_loader.py``). It lets
the REAL 8-GPU FSDP SFT stack consume the EXACT reference rank-0-fanout step-0
batch (256 frames) + the shared seed-4242 flow noise/time, so the production
step-0 loss/grad can be gated directly against the reference (AC-3, Gap 1 ->
``docs/evidence/phase6_ac3_production_stack_controlled.json``).

The dumper writes, for the single step-0 global batch:

* ``ref_step0_batch.npz`` — ``m{i}__{surface}`` for fanout chunk ``i`` in
  ``0..world_size-1`` (each surface a ``(micro, ...)`` array);
* ``step0_noise_time.npz`` — ``noise`` ``(world_size*micro, ah, ad)`` and
  ``time`` ``(world_size*micro,)`` drawn with numpy ``RandomState(4242)``.

The pinned loader expects (for ``n_steps == 1``):

* batches: each ``surface`` as ``(world_size, micro, ...)`` (flat index
  ``s*world_size + c``; rank ``r`` reads ``arr[r::world_size]``);
* noise/time: ``(n_steps, world_size*micro, ...)`` (rank ``r`` reads
  ``[r*micro:(r+1)*micro]``).

So we stack the ``world_size`` chunks and add a leading step axis. The result is
verified bit-for-bit against the committed 2x2 ``ref_batch_content_hash`` so the
pinned inputs are provably the same data fed through both models in the 2x2.

    python tools/build_pinned_step0_npz.py <dump_dir> <out_dir> [--world-size 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TWO2_JSON = os.path.join(_REPO, "docs/evidence/phase6_ac3_cross_feed_2x2.json")


def _content_hash(arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def main(dump_dir: str, out_dir: str, world_size: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    batch_npz = np.load(os.path.join(dump_dir, "ref_step0_batch.npz"))
    n_micro = int(batch_npz["n_micro"])
    if n_micro != world_size:
        raise ValueError(
            f"dump n_micro={n_micro} != world_size={world_size}; the dump topology "
            "does not match the intended run."
        )
    # surfaces present on chunk 0 (drop the bookkeeping n_micro key).
    surfaces = sorted(
        k.split("__", 1)[1] for k in batch_npz.files if k.startswith("m0__")
    )
    # stack the world_size fanout chunks -> (world_size, micro, ...).
    pinned = {
        s: np.stack([batch_npz[f"m{i}__{s}"] for i in range(world_size)], axis=0)
        for s in surfaces
    }

    nt = np.load(os.path.join(dump_dir, "step0_noise_time.npz"))
    noise = nt["noise"][None, ...]  # (1, world_size*micro, ah, ad)
    time = nt["time"][None, ...]  # (1, world_size*micro)

    # bit-identity gate: the pinned state+actions must equal the committed 2x2
    # reference batch (the exact data fed through both models in the 2x2).
    expect = json.load(open(_TWO2_JSON))["ref_batch_content_hash"]
    got = _content_hash(
        [pinned["state"][r] for r in range(world_size)]
        + [pinned["actions"][r] for r in range(world_size)]
    )
    if got != expect:
        raise SystemExit(
            f"pinned content hash {got} != committed 2x2 ref_batch_content_hash "
            f"{expect}; the dump is not the 2x2 reference batch."
        )

    np.savez(os.path.join(out_dir, "pinned_batches.npz"), **pinned)
    np.savez(
        os.path.join(out_dir, "pinned_noise_time.npz"), noise=noise, time=time
    )
    print(
        json.dumps(
            {
                "ok": True,
                "world_size": world_size,
                "micro": int(pinned["actions"].shape[1]),
                "global_batch": world_size * int(pinned["actions"].shape[1]),
                "surfaces": surfaces,
                "content_hash_matches_2x2_ref_batch": True,
                "out": out_dir,
            }
        )
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", help="dir holding the 2x2 dumper's npz outputs")
    ap.add_argument("out_dir", help="dir to write pinned_batches.npz + pinned_noise_time.npz")
    ap.add_argument("--world-size", type=int, default=8)
    args = ap.parse_args()
    main(args.dump_dir, args.out_dir, args.world_size)
