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

"""Direct parity vs the reference ``create_behavior_data_loader_torch``.

The reference (``openpi-comet-pytorch-mixed``) requires Python 3.11, so it is run
in its own venv via the subprocess dumper ``_ref_loader_dump.py`` (which builds
the real ``create_behavior_data_loader_torch`` for the ``turning_on_radio``
config on the local BEHAVIOR data and dumps a transformed batch + tokenizer
outputs). This test then compares the new ``openpi_pytorch`` loader/tokenizer:

* EXACT tokenizer parity (frame-independent): the new vendored
  ``PaligemmaTokenizer`` must produce byte-identical ids+mask to the reference
  tokenizer for fixed ``(prompt, state)`` inputs.
* Reference-loader output-contract parity: the new loader's first batch must
  match the real reference batch's shapes, normalize-then-pad layout (env dims in
  ``[:23]``, exact-zero pad tail), token length, image resolution, and value
  bands. (Exact frame-value parity is not asserted: the two streaming loaders are
  independently seeded, so they do not stream the same frame; exact normalize+pad
  values on a controlled input are covered by ``test_openpi_pytorch_sft_parity``.)

Skip-gated when the reference venv / source / data is unavailable.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import numpy as np
import pytest

_REF_PY = "/mnt/public/xzxuan/repos/openpi-comet/.venv/bin/python"
_DUMP = pathlib.Path(__file__).parent / "_ref_loader_dump.py"
_DATA = "/mnt/public/xzxuan/data/2025-challenge-demos"
_ASSETS_DIR = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"

# Must match _ref_loader_dump.py's _TOK_INPUTS exactly.
_TOK_INPUTS = [
    ("turning on radio", np.linspace(-1.0, 1.0, 23, dtype=np.float32)),
    ("turn on the radio", np.zeros(23, dtype=np.float32)),
    ("pick up radio from coffee table", np.full(23, 0.3, dtype=np.float32)),
]


@pytest.fixture(scope="module")
def ref_dump(tmp_path_factory):
    if not os.path.exists(_REF_PY):
        pytest.skip("reference py3.11 venv not available")
    if not os.path.isdir(_DATA):
        pytest.skip("BEHAVIOR data not available")
    out = tmp_path_factory.mktemp("ref")
    proc = subprocess.run(
        [_REF_PY, str(_DUMP), str(out)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    lines = [
        ln
        for ln in (proc.stdout + "\n" + proc.stderr).splitlines()
        if ln.startswith("REF_DUMP_RESULT ")
    ]
    if not lines:
        pytest.skip(f"reference dump produced no result; stderr tail: {proc.stderr[-300:]}")
    result = json.loads(lines[-1].split("REF_DUMP_RESULT ", 1)[1])
    return out, result


def test_tokenizer_exact_parity_vs_reference(ref_dump):
    out, result = ref_dump
    if not result.get("tokenizer_ok"):
        pytest.skip(f"reference tokenizer dump failed: {result.get('tokenizer_err')}")

    from rlinf.models.embodiment.openpi_pytorch.tokenizer import PaligemmaTokenizer

    ref = np.load(out / "ref_tokenizer.npz")
    tok = PaligemmaTokenizer(max_len=200)
    for i, (prompt, state) in enumerate(_TOK_INPUTS):
        ids, mask = tok.tokenize(prompt, state)
        np.testing.assert_array_equal(np.asarray(ids, dtype=np.int64), ref["ids"][i])
        np.testing.assert_array_equal(np.asarray(mask, dtype=bool), ref["mask"][i])


def test_fixed_sample_transform_parity_vs_reference(ref_dump):
    # The strongest gate: feed the SAME raw BEHAVIOR frame the reference
    # transformed through the new BehaviorSftTransform and assert exact (within
    # tolerance) parity of tokenized prompt+mask, the three resized images, and
    # the normalized+padded state and actions. This sidesteps streaming
    # non-determinism by comparing the transforms on one identical raw input.
    out, result = ref_dump
    if not result.get("sample_ok"):
        pytest.skip(f"reference fixed-sample dump failed: {result.get('sample_err')}")

    from rlinf.models.embodiment.openpi_pytorch.dataconfig.behavior_sft_transform import (
        BehaviorSftTransform,
        transform_behavior_sft_item,
    )
    from rlinf.models.embodiment.openpi_pytorch.normalize import load_norm_stats
    from rlinf.models.embodiment.openpi_pytorch.tokenizer import PaligemmaTokenizer

    raw = np.load(out / "ref_raw_frame.npz", allow_pickle=True)
    ref = np.load(out / "ref_item.npz", allow_pickle=True)

    norm_stats = load_norm_stats(
        pathlib.Path(_ASSETS_DIR) / "behavior-1k" / "2025-challenge-demos"
    )
    transform = BehaviorSftTransform(
        norm_stats=norm_stats,
        action_dim=32,
        max_token_len=200,
        tokenizer=PaligemmaTokenizer(max_len=200),
    )
    frame = {
        "observation.images.rgb.head": raw["head"],
        "observation.images.rgb.left_wrist": raw["left"],
        "observation.images.rgb.right_wrist": raw["right"],
        "observation.state": raw["state"],
        "action": raw["action"],
        "task": str(raw["task"]),
    }
    item = transform_behavior_sft_item(frame, transform)

    # Normalized + padded state and actions match the reference within tolerance.
    np.testing.assert_allclose(
        np.asarray(item["state"]), ref["state"], rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(item["actions"]), ref["actions"], rtol=1e-4, atol=1e-4
    )
    # Tokenized discrete-state prompt + mask are byte-identical.
    np.testing.assert_array_equal(
        np.asarray(item["tokenized_prompt"]), ref["tokenized_prompt"]
    )
    np.testing.assert_array_equal(
        np.asarray(item["tokenized_prompt_mask"]), ref["tokenized_prompt_mask"]
    )
    # The three resized images match the reference pixels (within rounding).
    for cam, key in (
        ("base", "base_0_rgb"),
        ("left", "left_wrist_0_rgb"),
        ("right", "right_wrist_0_rgb"),
    ):
        new_img = np.asarray(item["image"][key])
        assert new_img.shape == tuple(ref[cam].shape)
        np.testing.assert_allclose(
            new_img.astype(np.int16), ref[cam].astype(np.int16), atol=2
        )


def test_loader_output_contract_parity_vs_reference(ref_dump):
    out, result = ref_dump
    if not result.get("loader_ok"):
        pytest.skip(f"reference loader dump failed: {result.get('loader_err')}")

    from rlinf.models.embodiment.openpi_pytorch.dataconfig import (
        create_behavior_sft_data_loader,
    )

    ref = np.load(out / "ref_batch.npz", allow_pickle=True)
    # Reference loader output contract: normalize-then-pad to 32, tokens 200, img 224.
    assert ref["state"].shape[1] == 32
    assert ref["actions"].shape[1:] == (32, 32)
    assert ref["tokenized_prompt"].shape[1] == 200
    assert ref["image"].shape[-2:] == (224, 224)
    np.testing.assert_array_equal(ref["state"][:, 23:], 0.0)
    np.testing.assert_array_equal(ref["actions"][..., 23:], 0.0)

    loader = create_behavior_sft_data_loader(
        behavior_dataset_root=_DATA,
        assets_dir=_ASSETS_DIR,
        asset_id="behavior-1k/2025-challenge-demos",
        repo_id="behavior-1k/2025-challenge-demos",
        tasks=["turning_on_radio"],
        action_dim=32,
        action_horizon=32,
        max_token_len=200,
        batch_size=2,
        num_workers=0,
        seed=0,
    )
    observation, actions = next(iter(loader))
    new_state = observation.state.numpy()
    new_actions = actions.numpy()
    new_image = next(iter(observation.images.values())).numpy()

    # Same output contract as the real reference loader.
    assert new_state.shape[1] == 32
    assert new_actions.shape[1:] == (32, 32)
    assert observation.tokenized_prompt.shape[1] == 200
    assert new_image.shape[1:3] == (224, 224)  # (B, H, W, C)
    np.testing.assert_array_equal(new_state[:, 23:], 0.0)
    np.testing.assert_array_equal(new_actions[..., 23:], 0.0)

    # Normalized env-dim values lie in the same band as the reference's, and the
    # images are float in [-1, 1] on both sides.
    assert new_image.min() >= -1.0001 and new_image.max() <= 1.0001
    assert float(ref["image"].min()) >= -1.0001 and float(ref["image"].max()) <= 1.0001
    for ref_arr, new_arr in (
        (ref["state"][:, :23], new_state[:, :23]),
        (ref["actions"][..., :23], new_actions[..., :23]),
    ):
        assert np.isfinite(new_arr).all()
        assert new_arr.min() >= ref_arr.min() - 1.0
        assert new_arr.max() <= ref_arr.max() + 1.0
