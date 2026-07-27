# Copyright 2026 The RLinf Authors.
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

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi_pytorch.eval_action_model import (
    OpenPiPytorchEvalActionModel,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.sft_action_model import (
    OpenPiPytorchSFTActionModel,
)
from rlinf.models.embodiment.openpi_pytorch.transforms_pipeline import (
    build_openpi_transforms,
    find_subtask_tokenizer,
)
from rlinf.models.embodiment.openpi_pytorch.utils import tokenizer as tokenizer_module
from rlinf.models.embodiment.openpi_pytorch.utils.model_builders import (
    _resolve_transform_kwargs,
)
from rlinf.models.embodiment.openpi_pytorch.utils.tokenizer import (
    PaligemmaTokenizer,
    TokenizeSubtaskPrompt,
)

_TOKENIZER_PATH = Path(
    "/mnt/public/xzxuan/models/paligemma_tokenizer/paligemma_tokenizer.model"
)
_EVAL_CHECKPOINT = Path(
    "/mnt/public/xzxuan/repos/RLinf/outputs/sft_ckpt/vlm_vla_50k_sg_false_no_state"
)


class _FakeSubtaskTokenizer:
    eos_token_id = 2

    def tokenize_with_subtask(self, prompt, state, response):
        del prompt, state, response
        return (
            np.asarray([1, 2]),
            np.asarray([True, True]),
            np.asarray([False, True]),
            np.asarray([False, True]),
            np.asarray([True, False]),
        )

    def decode(self, token_ids):
        return "decoded:" + ",".join(map(str, token_ids))


class _DummyPi0(torch.nn.Module):
    action_dim = 32
    action_horizon = 2

    def __init__(self, output):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.output = output
        self.vlm_vla = isinstance(output, tuple)

    def compute_loss(self, observation, actions, train):
        del observation, actions, train
        return self.output

    def reason_and_sample_actions(self, observation, **kwargs):
        del observation, kwargs
        actions, generation = self.output
        return actions.to(self.anchor.device), generation


def _observation() -> Observation:
    return Observation(
        images={},
        image_masks={},
        state=torch.zeros(1, 32),
        tokenized_prompt=torch.ones(1, 2, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(1, 2, dtype=torch.bool),
        token_ar_mask=torch.tensor([[False, True]]),
        token_loss_mask=torch.tensor([[False, True]]),
        token_kv_cache_mask=torch.tensor([[True, False]]),
    )


def test_subtask_transform_emits_all_masks_and_consumes_response():
    transform = TokenizeSubtaskPrompt(_FakeSubtaskTokenizer(), inject_state=False)
    output = transform(
        {
            "prompt": "main task",
            "response": "local skill",
            "state": np.zeros(32),
        }
    )

    assert "prompt" not in output
    assert "response" not in output
    assert output["tokenized_prompt"].tolist() == [1, 2]
    assert output["tokenized_prompt_mask"].tolist() == [True, True]
    assert output["token_ar_mask"].tolist() == [False, True]
    assert output["token_loss_mask"].tolist() == [False, True]
    assert output["token_kv_cache_mask"].tolist() == [True, False]


@pytest.mark.skipif(
    not _TOKENIZER_PATH.is_file(), reason="local PaliGemma tokenizer unavailable"
)
def test_real_subtask_tokenizer_masks_eos_out_of_action_cache(monkeypatch):
    downloads = []

    def _use_local_tokenizer(uri, **kwargs):
        downloads.append((uri, kwargs))
        return _TOKENIZER_PATH

    monkeypatch.setattr(
        tokenizer_module.download, "maybe_download", _use_local_tokenizer
    )
    tokenizer = PaligemmaTokenizer(max_len=64)
    tokens, valid, ar, loss, kv = tokenizer.tokenize_with_subtask(
        "turning_on_radio",
        None,
        "press the radio button",
    )

    assert tokens.shape == valid.shape == ar.shape == loss.shape == kv.shape == (64,)
    eos_index = int(np.flatnonzero(tokens == tokenizer.eos_token_id)[-1])
    assert valid[eos_index] and ar[eos_index] and loss[eos_index]
    assert not kv[eos_index]
    assert not valid[eos_index + 1 :].any()
    assert downloads == [
        (
            "gs://big_vision/paligemma_tokenizer.model",
            {"gs": {"token": "anon"}},
        )
    ]


@pytest.mark.skipif(
    not _TOKENIZER_PATH.is_file(), reason="local PaliGemma tokenizer unavailable"
)
def test_action_prompt_tokenization_matches_official_openpi(monkeypatch):
    monkeypatch.setattr(
        tokenizer_module.download,
        "maybe_download",
        lambda uri, **kwargs: (
            _TOKENIZER_PATH
            if uri == "gs://big_vision/paligemma_tokenizer.model"
            else Path(uri)
        ),
    )
    from openpi.models.tokenizer import PaligemmaTokenizer as OpenPiTokenizer

    tokenizer = PaligemmaTokenizer(max_len=64)
    reference = OpenPiTokenizer(max_len=64)
    state = np.linspace(-0.9, 0.9, 5)

    for prompt_state in (None, state):
        actual = tokenizer.tokenize("Turn_on_the_radio\nnow", prompt_state)
        expected = reference.tokenize("Turn_on_the_radio\nnow", prompt_state)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])


def test_sft_wrapper_preserves_vlm_loss_dict():
    metrics = {
        "loss": torch.tensor(3.0),
        "action_loss": torch.tensor(2.0),
        "language_loss": torch.tensor(1.0),
        "language_acc": torch.tensor(0.5),
    }
    wrapper = OpenPiPytorchSFTActionModel(
        _DummyPi0(metrics), num_steps=5, action_env_dim=23
    )

    output = wrapper.sft_forward((_observation(), torch.zeros(1, 2, 32)))

    assert output is metrics


def test_eval_wrapper_runs_reason_then_action_and_decodes_generation():
    # Production eval runs the model in bf16; NumPy-backed OpenPI output
    # transforms must cross that boundary through float32.
    model_actions = torch.zeros(1, 2, 32, dtype=torch.bfloat16)
    generation = {
        "tokens": torch.tensor([[7, 2]]),
        "eos_steps": torch.tensor([1]),
        "terminated": torch.tensor([True]),
    }
    wrapper = OpenPiPytorchEvalActionModel(
        _DummyPi0((model_actions, generation)),
        num_steps=5,
        action_env_dim=23,
        action_chunk=2,
        subtask_tokenizer=_FakeSubtaskTokenizer(),
    )
    wrapper.setup_wrappers([], [])

    actions, result = wrapper._predict_eval(_observation(), noise=None, rng=None)

    assert actions.shape == (1, 2, 32)
    assert actions.dtype == torch.float32
    assert result["generated_text"] == ["decoded:7"]
    assert result["generation_terminated"].tolist() == [True]
    assert result["forward_inputs"]["generated_token_ids"].tolist() == [[7, 2]]


@pytest.mark.skipif(
    not (_TOKENIZER_PATH.is_file() and _EVAL_CHECKPOINT.is_dir()),
    reason="local OpenPI eval assets unavailable",
)
def test_shared_pipeline_replaces_only_prompt_tokenizer_for_vlm_vla(monkeypatch):
    monkeypatch.setattr(
        tokenizer_module.download,
        "maybe_download",
        lambda uri, **kwargs: (
            _TOKENIZER_PATH
            if uri == "gs://big_vision/paligemma_tokenizer.model"
            else Path(uri)
        ),
    )
    input_transforms, output_transforms = build_openpi_transforms(
        str(_EVAL_CHECKPOINT),
        "pi05_behavior",
        mode="vlm_vla",
        state_token="none",
        action_env_dim=23,
        max_token_len=200,
    )

    tokenizer = find_subtask_tokenizer(input_transforms)
    assert tokenizer is not None
    assert tokenizer.max_len == 200
    assert any(
        type(transform).__name__ == "Normalize" for transform in input_transforms
    )
    assert any(
        type(transform).__name__ == "ResizeImages" for transform in input_transforms
    )
    assert any(
        type(transform).__name__ == "PadStatesAndActions"
        for transform in input_transforms
    )
    assert any(
        type(transform).__name__ == "Unnormalize" for transform in output_transforms
    )

    vla_input_transforms, _ = build_openpi_transforms(
        str(_EVAL_CHECKPOINT),
        "pi05_behavior",
        mode="vla",
        state_token="none",
        action_env_dim=23,
        max_token_len=200,
    )
    vla_prompt_transform = next(
        transform
        for transform in vla_input_transforms
        if type(transform).__name__ == "TokenizePrompt"
    )
    assert isinstance(vla_prompt_transform.tokenizer, PaligemmaTokenizer)


def test_required_vlm_configs_use_generic_state_and_data_fields():
    repo_root = Path(__file__).resolve().parents[2]
    single_path = repo_root / "examples/sft/config/behavior_pi05_vlm_vla.yaml"
    fifty_path = repo_root / "examples/sft/config/behavior_50tasks_pi05_vlm_vla.yaml"
    eval_path = (
        repo_root
        / "evaluations/behavior/behavior_openpi_pi05_pytorch_vlm_vla_eval.yaml"
    )

    single = OmegaConf.load(single_path)
    fifty = OmegaConf.load(fifty_path)
    eval_cfg = OmegaConf.load(eval_path)

    for cfg in (single, fifty):
        assert cfg.actor.model.openpi.mode == "vlm_vla"
        assert cfg.actor.model.openpi.state_token == "none"
        assert cfg.actor.model.model_path == (
            "/mnt/public/xzxuan/models/pi05_base_pytorch_new"
        )
    assert len(fifty.data.tasks) == 50
    assert fifty.actor.model.openpi.max_token_len == 288
    assert eval_cfg.rollout.model.openpi.state_token == "none"
    assert eval_cfg.rollout.model.model_path == str(_EVAL_CHECKPOINT)

    for path in (single_path, fifty_path, eval_path):
        text = path.read_text()
        assert "paligemma_tokenizer" not in text
        assert "discrete_state_input" not in text
        assert "train_data_paths_" not in text
        assert "behavior_dataset_root_" not in text
        assert "assets_dir_" not in text
        assert "asset_id_" not in text


def test_legacy_discrete_state_selector_remains_compatible():
    state_free = _resolve_transform_kwargs(
        OmegaConf.create({"discrete_state_input": False})
    )
    upstream_default = _resolve_transform_kwargs(OmegaConf.create({}))

    assert state_free["state_token"] == "none"
    assert upstream_default["state_token"] == "abs_joint_old"
