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

"""Dump a fixed batch of REAL BEHAVIOR eval observations for behavior comparison.

Constructs the SAME ``BehaviorEnv`` that ``get_env_cls`` returns for the BEHAVIOR
``use_skill:false`` eval path (the OmniGibson-backed env, ``env/behavior_r1pro``
config: task ``turning_on_radio``), calls ``reset()`` ONCE, and serializes the raw
``env_obs`` the eval policy consumes -- ``main_images`` / ``wrist_images`` /
``states`` / ``task_descriptions`` -- plus a sha256 over each field and the
resolved env knobs (seed, num_envs, task). These REAL observations (not synthetic
random tensors) are the fixed set replayed offline through both trained checkpoints.

Run with the embodied venv + the eval OmniGibson env vars, e.g.::

    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMNIGIBSON_HEADLESS=1 \
    /mnt/public/xzxuan/.venv_pi/bin/python \
        tests/unit_tests/_behavior_eval_obs_dump.py <out_dir> <num_envs>

Writes ``eval_obs.npz`` + ``eval_obs_meta.json`` and prints one ``EVAL_OBS <json>``
line (hashes + knobs + provenance).
"""

import hashlib
import json
import os
import sys

import numpy as np
from omegaconf import OmegaConf

_ENV_CFG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "examples/embodiment/config/env/behavior_r1pro.yaml",
)
_SEED = 0  # the eval env seed (env.eval.seed); recorded with the obs.


def _np(x):
    import torch

    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _sha(arr) -> str:
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(a.tobytes()).hexdigest()


def main(out_dir, num_envs):
    os.makedirs(out_dir, exist_ok=True)
    result = {"ok": False}
    env = None
    try:
        from rlinf.envs.behavior.behavior_env import BehaviorEnv

        env_cfg = OmegaConf.load(_ENV_CFG)
        env_cfg.total_num_envs = num_envs
        env_cfg.seed = _SEED
        # One reset only; keep the boot minimal and deterministic-per-seed.
        env_cfg.num_env_subprocess = 1

        # record_metrics defaults to True; it must stay True because reset() calls
        # _record_metrics/_reset_metrics, which reference the metric buffers that
        # only _init_metrics (gated on record_metrics) allocates.
        env = BehaviorEnv(
            cfg=env_cfg,
            num_envs=num_envs,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
        obs, _ = env.reset()

        main_images = _np(obs["main_images"])
        wrist_images = _np(obs["wrist_images"])
        states = _np(obs["states"])
        tasks = [str(t) for t in obs["task_descriptions"]]

        np.savez(
            os.path.join(out_dir, "eval_obs.npz"),
            main_images=main_images,
            wrist_images=wrist_images,
            states=states,
        )
        hashes = {
            "main_images": _sha(main_images),
            "wrist_images": _sha(wrist_images),
            "states": _sha(states),
            "task_descriptions": hashlib.sha256(json.dumps(tasks).encode()).hexdigest(),
        }
        meta = {
            "num_envs": int(num_envs),
            "env_seed": _SEED,
            "env_type": str(env_cfg.env_type),
            "activity_name": str(
                OmegaConf.select(env_cfg, "omni_config.task.activity_name")
            ),
            "task_descriptions": tasks,
            "shapes": {
                "main_images": list(main_images.shape),
                "wrist_images": list(wrist_images.shape),
                "states": list(states.shape),
            },
            "dtypes": {
                "main_images": str(main_images.dtype),
                "wrist_images": str(wrist_images.dtype),
                "states": str(states.dtype),
            },
            "hashes": hashes,
        }
        with open(os.path.join(out_dir, "eval_obs_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        result.update(ok=True, **meta)
    except Exception as e:  # pragma: no cover - environment dependent
        import traceback

        result["err"] = f"{type(e).__name__}: {str(e)[:400]}"
        result["trace"] = traceback.format_exc()[-1500:]
    finally:
        if env is not None:
            try:
                env.env_close()
            except Exception:
                pass

    print("EVAL_OBS " + json.dumps(result))


if __name__ == "__main__":
    out = (
        sys.argv[1] if len(sys.argv) > 1 else "/mnt/public/xzxuan/tmp/rlcr_r7/eval_obs"
    )
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main(out, n)
