# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for RLinf RPent GRPO agent loop, runner, and Qwen3.5 convertor."""

from __future__ import annotations

import hashlib
import inspect
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rlinf.agents.rpent import (
    SimulatorBatchSampler,
    episode_task,
    merge_prefix_turns,
    mixed_simulators,
)
from rlinf.runners.rpent_runner import _aggregate_metrics
from rlinf.utils.convertor.utils import Qwen35Convertor
from rlinf.workers.agent.agent_loop import AgentLoopOutput


def test_episode_task_parsing():
    raw = '{"environment": "libero", "suite": "libero_goal", "task": 0, "seed": 1}'
    assert episode_task(raw)["environment"] == "libero"
    assert episode_task([raw])["seed"] == 1
    with pytest.raises(ValueError):
        episode_task([raw, raw])


def test_merge_prefix_turns_respects_chain_and_budget():
    t0 = AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        response_text="a",
        response_logprobs=[-0.1, -0.2],
        response_weight_versions=[0, 0],
        response_images={"k": 0},
        version_start=0,
        version_end=0,
        is_end=False,
        reward_score=1.0,
        extra_fields={"turn_id": 0, "chain_id": 0, "tool_names": ["view_env"]},
    )
    t1 = AgentLoopOutput(
        prompt_ids=[1, 2, 3, 4, 5, 99, 98],
        response_ids=[6, 7],
        response_text="b",
        response_logprobs=[-0.3, -0.4],
        response_weight_versions=[0, 0],
        response_images={"k": 1},
        version_start=0,
        version_end=0,
        is_end=False,
        reward_score=1.0,
        extra_fields={"turn_id": 1, "chain_id": 0, "tool_names": ["act"]},
    )
    t2 = AgentLoopOutput(
        prompt_ids=[1, 2, 77],
        response_ids=[8],
        response_text="c",
        response_logprobs=[-0.5],
        response_weight_versions=[0],
        response_images={"k": 2},
        version_start=0,
        version_end=0,
        is_end=True,
        reward_score=1.0,
        extra_fields={"turn_id": 2, "chain_id": 1, "tool_names": []},
    )
    merged = merge_prefix_turns([t0, t1, t2], max_scored_tokens=16)
    assert len(merged) == 2
    assert merged[0].prompt_ids == [1, 2, 3]
    assert merged[0].response_ids == [4, 5, 99, 98, 6, 7]
    assert merged[0].response_mask == [1, 1, 0, 0, 1, 1]
    assert merged[0].response_logprobs == [-0.1, -0.2, 0.0, 0.0, -0.3, -0.4]
    assert merged[0].response_images == {"k": 1}
    assert merged[0].extra_fields["turn_ids"] == [0, 1]
    assert merged[1].extra_fields["turn_ids"] == [2]

    # Budget split: max_scored_tokens=3 forces t0 and t1 into separate rows.
    split = merge_prefix_turns([t0, t1], max_scored_tokens=3)
    assert len(split) == 2


@pytest.mark.parametrize("second", ["robotwin", "metaworld", "maniskill"])
def test_mixed_simulators_and_sampler_quotas(second):
    cfg = OmegaConf.create(
        {
            "algorithm": {"group_size": 8},
            "data": {"rollout_batch_size": 2, "sampling_batch_size": 4},
            "env": {
                "rpent": {
                    "libero": {
                        "endpoint": "http://127.0.0.1:28561",
                        "train_data_paths": "/tmp/a.jsonl",
                    },
                    second: {
                        "endpoint": "http://127.0.0.1:28562",
                        "train_data_paths": "/tmp/b.jsonl",
                    },
                }
            },
        }
    )
    sims = mixed_simulators(cfg)
    assert set(sims) == {"libero", second}
    assert sims["libero"].rollout_groups == 1
    assert sims["libero"].sampling_groups == 2

    sampler = SimulatorBatchSampler([10, 10], [2, 2], shuffle=True, seed=42)
    batches = list(sampler)
    assert len(batches) == 5
    for batch in batches:
        assert len(batch) == 4
        assert sum(idx < 10 for idx in batch) == 2
        assert sum(idx >= 10 for idx in batch) == 2


def test_aggregate_metrics_and_qwen35_bf16_norm():
    agg = _aggregate_metrics(
        [
            {"__mean__/rpent/success_rate": (1, 1), "__sum__/rpent/episodes": 1},
            {"__mean__/rpent/success_rate": (0, 1), "__sum__/rpent/episodes": 1},
        ]
    )
    assert agg["rpent/success_rate"] == pytest.approx(0.5)
    assert agg["rpent/episodes"] == pytest.approx(2.0)

    import torch

    # Verify Qwen3.5 GDN out_norm weight adds +1.0 in fp32 before casting back to bf16
    compact_src = " ".join(inspect.getsource(Qwen35Convertor).split())
    assert "tensor.float() + 1.0" in compact_src
    assert ").to(tensor.dtype)" in compact_src
    conv = object.__new__(Qwen35Convertor)
    conv.convert_rules = []
    t = torch.tensor([0.001953125, -0.5, 0.25], dtype=torch.bfloat16)
    out = Qwen35Convertor.convert(
        conv, {"decoder.layers.0.self_attention.out_norm.weight": t}
    )
    got = out["model.language_model.layers.0.linear_attn.norm.weight"]
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, (t.float() + 1.0).to(torch.bfloat16))


@pytest.mark.parametrize("second", ["robotwin", "metaworld", "maniskill"])
def test_validate_sim_pool_capacity(second):
    from rlinf.agents.rpent.simulators import SimulatorSpec
    from rlinf.runners.rpent_runner import validate_sim_pool_capacity

    sims = {
        "libero": SimulatorSpec(
            "libero", "http://127.0.0.1:28561", "/tmp/a.jsonl", 1, 2
        ),
        second: SimulatorSpec(second, "http://127.0.0.1:28562", "/tmp/b.jsonl", 1, 2),
    }

    # Exact capacity succeeds with uncapped client
    statuses = validate_sim_pool_capacity(
        simulators=sims,
        default_endpoint=None,
        sampling_groups=4,
        group_size=8,
        num_agent_workers=4,
        episodes_per_worker=None,
        fetch_health=lambda ep: {"ok": True, "workers_total": 16, "workers_idle": 16},
    )
    assert statuses["libero"]["workers_total"] == 16
    assert statuses[second]["required"] == 16

    # Undersized simulator pool fails fast
    with pytest.raises(RuntimeError, match="workers_total=10"):
        validate_sim_pool_capacity(
            simulators=sims,
            default_endpoint=None,
            sampling_groups=4,
            group_size=8,
            num_agent_workers=4,
            episodes_per_worker=None,
            fetch_health=lambda ep: {
                "ok": True,
                "workers_total": 10,
                "workers_idle": 10,
            },
        )

    # Throttling client episodes_per_worker fails fast
    with pytest.raises(RuntimeError, match="episodes_per_worker=4"):
        validate_sim_pool_capacity(
            simulators=sims,
            default_endpoint=None,
            sampling_groups=4,
            group_size=8,
            num_agent_workers=4,
            episodes_per_worker=4,
            fetch_health=lambda ep: {
                "ok": True,
                "workers_total": 16,
                "workers_idle": 16,
            },
        )


def test_zero_base64_conversation_history_and_local_staging():
    from rlinf.agents.rpent.agent_loop import (
        _BINARY_EXECUTE_MEDIA_TYPE,
        _unpack_execute_response,
    )
    from rlinf.agents.rpent.history import AppendOnlyHistory, EpisodeConversation
    from rlinf.hybrid_engines.sglang.multimodal.replay import stage_local_images

    class _FakeInnerTok:
        def convert_tokens_to_ids(self, tok: str) -> int:
            return 999 if tok == "<|image_pad|>" else 1

        def __call__(self, text: str, add_special_tokens: bool = False):
            # Map each <|image_pad|> marker to token 999, words to dummy tokens
            ids = []
            for part in text.split():
                ids.append(
                    999 if "<|image_pad|>" in part else (abs(hash(part)) % 100 + 10)
                )
            return {"input_ids": ids}

    class _FakeAgenticTok:
        def __init__(self):
            self.tokenizer = _FakeInnerTok()

        def apply_chat_template(
            self, messages, tools=None, tokenize=False, add_generation_prompt=True
        ):
            chunks = []
            for m in messages:
                c = m.get("content", "")
                if isinstance(c, list):
                    for part in c:
                        if part.get("type") == "image_url":
                            chunks.append("<|image_pad|>")
                        else:
                            chunks.append(part.get("text", ""))
                else:
                    chunks.append(str(c))
            return " ".join(chunks)

        def render_incremental(self, delta_msgs):
            text = self.apply_chat_template(delta_msgs)
            return self.tokenizer(text)["input_ids"]

        def join_increment(self, prefix_ids, increment):
            return [*prefix_ids, *increment]

    png1 = b"\x89PNG\r\n\x1a\nfake1"
    png2 = b"\x89PNG\r\n\x1a\nfake2"
    conv = EpisodeConversation(
        system_prompt="SYS",
        user_message="USR",
        initial_observation_text="OBS0",
        tools=[],
        environment="libero",
    )
    hist = AppendOnlyHistory(_FakeAgenticTok())

    # Turn 0 (no images)
    m0 = conv.wire_messages()
    r0 = hist.render_turn(m0, conv.tools, conv.images())
    assert r0.images == []
    a0 = {
        "role": "assistant",
        "content": "call",
        "tool_calls": [
            {
                "id": "c0",
                "type": "function",
                "function": {"name": "view_env", "arguments": "{}"},
            }
        ],
    }
    hist.record(m0, conv.tools, r0.input_ids, [42, 43], a0)
    conv.append_assistant(a0)
    conv.append_tool_result(
        tool_name="view_env", tool_call_id="c0", text="ok", images=[png1]
    )

    # Turn 1 (continues prefix, zero base64, raw bytes preserved by identity)
    m1 = conv.wire_messages()
    assert all("base64" not in str(m) for m in m1)
    r1 = hist.render_turn(m1, conv.tools, conv.images())
    assert r1.images == [png1]
    assert r1.images[0] is png1
    assert len(r1.image_spans) == 1
    assert hist.chain_id == 1

    # Turn 2 (adds second image, reuses first raw bytes without copy or decode)
    a1 = {
        "role": "assistant",
        "content": "call2",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "view_env", "arguments": "{}"},
            }
        ],
    }
    hist.record(m1, conv.tools, r1.input_ids, [44], a1)
    conv.append_assistant(a1)
    conv.append_tool_result(
        tool_name="view_env", tool_call_id="c1", text="ok2", images=[png2]
    )
    r2 = hist.render_turn(conv.wire_messages(), conv.tools, conv.images())
    assert r2.images == [png1, png2]
    assert r2.images[0] is png1 and r2.images[1] is png2
    assert hist.chain_id == 1

    # Local staging writes once and returns stable .png file paths
    staged_a = stage_local_images([png1, png2])
    staged_b = stage_local_images([png1, png2])
    assert staged_a == staged_b
    assert all(
        isinstance(p, str) and p.endswith(".png") and Path(p).read_bytes() == raw
        for p, raw in zip(staged_a, [png1, png2])
    )

    # Binary execute unpacker sanity
    import json
    import struct

    meta = {"text": "hi", "done": False, "success": False, "truncated": False}
    meta_b = json.dumps(meta).encode()
    buf = (
        struct.pack("!I", len(meta_b))
        + meta_b
        + struct.pack("!I", 1)
        + struct.pack("!I", len(png1))
        + png1
    )
    got_meta, got_imgs = _unpack_execute_response(buf)
    assert got_meta == meta and got_imgs == [png1]
    assert _BINARY_EXECUTE_MEDIA_TYPE.startswith("application/")


def test_validate_sim_pool_capacity_48gpu_scale() -> None:
    """Verify 48-GPU 384-episode (192 LIBERO + 192 RoboTwin) capacity validation."""
    from rlinf.agents.rpent.simulators import SimulatorSpec
    from rlinf.runners.rpent_runner import validate_sim_pool_capacity

    sims = {
        "libero": SimulatorSpec("libero", "http://127.0.0.1:28561", "a.jsonl", 16, 24),
        "robotwin": SimulatorSpec(
            "robotwin", "http://127.0.0.1:28562", "b.jsonl", 16, 24
        ),
    }
    res = validate_sim_pool_capacity(
        simulators=sims,
        default_endpoint=None,
        sampling_groups=48,
        group_size=8,
        num_agent_workers=32,
        episodes_per_worker=None,
        fetch_health=lambda _ep: {"workers_total": 192, "workers_idle": 192},
    )
    assert res["libero"]["required"] == 192
    assert res["robotwin"]["required"] == 192


def test_rpent_runner_pool_sleep_wake_lifecycle(monkeypatch) -> None:
    """Verify RPentGRPORunner._pool_lifecycle posts /pool/{sleep,wake} to all endpoints."""
    import httpx
    from omegaconf import OmegaConf

    from rlinf.agents.rpent.simulators import SimulatorSpec
    from rlinf.runners.rpent_runner import RPentGRPORunner

    calls: list[str] = []

    class _FakeResp:
        def __init__(self, url: str):
            self.url = url

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            sleeping = self.url.endswith("/sleep")
            return {
                "ok": True,
                "environment": "libero" if "28561" in self.url else "robotwin",
                "workers_target": 192,
                "workers_total": 32 if sleeping else 192,
                "sleeping": sleeping,
            }

    def _fake_post(url, **kwargs):
        calls.append(url)
        return _FakeResp(url)

    monkeypatch.setattr(httpx, "post", _fake_post)

    runner = object.__new__(RPentGRPORunner)
    runner.simulators = {
        "libero": SimulatorSpec("libero", "http://127.0.0.1:28561", "a.jsonl", 16, 24),
        "robotwin": SimulatorSpec(
            "robotwin", "http://127.0.0.1:28562", "b.jsonl", 16, 24
        ),
    }
    runner.cfg = OmegaConf.create(
        {"tools": {"rpent": {"sleep_sim_pools_during_training": True}}}
    )

    assert runner._sleep_pools_enabled() is True
    runner._pool_lifecycle("sleep")
    runner._pool_lifecycle("wake")
    assert set(calls) == {
        "http://127.0.0.1:28561/pool/sleep",
        "http://127.0.0.1:28562/pool/sleep",
        "http://127.0.0.1:28561/pool/wake",
        "http://127.0.0.1:28562/pool/wake",
    }


def test_sglang_multimodal_double_retraction_and_idle_pause(monkeypatch) -> None:
    """Verify multi-retract M-RoPE slicing, late-request parking, and idle pause."""
    pytest.importorskip("sglang")
    from types import SimpleNamespace

    import torch
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    from rlinf.hybrid_engines.sglang.common.sgl_scheduler import Scheduler, _Scheduler

    class _Batch:
        def __init__(self, reqs):
            self.reqs = reqs

        def filter_batch(self, keep_indices):
            self.reqs = [self.reqs[i] for i in keep_indices]

    req = Req(
        "mm-active",
        "",
        [1, 2, 3, 4],
        SamplingParams(max_new_tokens=20),
        return_logprob=True,
    )
    req.output_ids = [10, 11]
    req.output_token_logprobs_val = [-0.2, -0.3]
    req.logprob_start_len = 4
    # Simulate an already-extended M-RoPE tensor from a prior retraction (4 + 2 = 6 cols).
    mm_item = SimpleNamespace(
        feature=torch.zeros(2, 4), precomputed_embeddings=None, image_grid_thw=None
    )
    req.multimodal_inputs = SimpleNamespace(
        mrope_positions=torch.arange(6).unsqueeze(0).expand(3, -1).clone(),
        mm_items=[mm_item],
        vision_position_ids=None,
        visible_frame_counts=None,
    )
    late = Req(
        "mm-late", "", [5, 6], SamplingParams(max_new_tokens=20), return_logprob=True
    )

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._parked_requests = None
    scheduler.waiting_queue = []
    scheduler.running_batch = _Batch([req])
    scheduler.last_batch = scheduler.cur_batch = scheduler.chunked_req = None
    scheduler.enable_overlap = False
    scheduler._engine_paused = True
    scheduler.model_config = SimpleNamespace(is_multimodal=False)
    scheduler.offloaded_tags = dict.fromkeys(
        ["weights", "kv_cache", "cuda_graph"], False
    )
    scheduler.is_weight_offloaded = False
    scheduler.weight_reload = "sync"
    scheduler._rlinf_worker = SimpleNamespace(
        torch_platform=SimpleNamespace(synchronize=lambda: None)
    )

    def _retract(self, _request):
        for r in list(self.running_batch.reqs):
            r.reset_for_retract()
            self._add_request_to_queue(r)
        self.running_batch.filter_batch(keep_indices=[])

    monkeypatch.setattr(_Scheduler, "pause_generation", _retract)
    monkeypatch.setattr(
        _Scheduler,
        "flush_cache",
        lambda self: not self.waiting_queue
        and not self.running_batch.reqs
        and self.chunked_req is None,
    )
    monkeypatch.setattr(
        _Scheduler,
        "_add_request_to_queue",
        lambda self, r, is_retracted=False: self.waiting_queue.append(r),
    )
    monkeypatch.setattr(
        _Scheduler,
        "is_fully_idle",
        lambda self: not self.waiting_queue and not self.running_batch.reqs,
    )
    monkeypatch.setattr(
        _Scheduler,
        "release_memory_occupation",
        lambda self, _r: None
        if self.is_fully_idle()
        else (_ for _ in ()).throw(AssertionError("not idle")),
    )
    monkeypatch.setattr(_Scheduler, "resume_memory_occupation", lambda self, _r: None)

    # Cycle 1: active request parked; M-RoPE sliced back to prompt_len=4; late request parked.
    scheduler.release_memory_occupation(
        SimpleNamespace(tags=["weights", "kv_cache", "cuda_graph"])
    )
    assert req.multimodal_inputs.mrope_positions.shape == (3, 4)
    assert req.logprob_start_len == -1
    scheduler._add_request_to_queue(late)
    assert set(scheduler._parked_requests) == {"mm-active", "mm-late"}

    scheduler.resume_memory_occupation(SimpleNamespace(tags=["weights"]))
    assert scheduler.waiting_queue == []
    scheduler.resume_memory_occupation(SimpleNamespace(tags=["kv_cache", "cuda_graph"]))
    assert scheduler.waiting_queue == [req, late]

    # Cycle 2: engine is idle at pause time -> late request arriving while offloaded is still parked!
    scheduler.waiting_queue.clear()
    scheduler.release_memory_occupation(
        SimpleNamespace(tags=["weights", "kv_cache", "cuda_graph"])
    )
    assert scheduler._parked_requests == {}
    scheduler._add_request_to_queue(late)
    assert scheduler.waiting_queue == []
    assert list(scheduler._parked_requests) == ["mm-late"]
    scheduler.resume_memory_occupation(
        SimpleNamespace(tags=["weights", "kv_cache", "cuda_graph"])
    )
    assert scheduler.waiting_queue == [late]


def test_trajectory_level_replica_affinity_balances_evenly() -> None:
    """Verify deterministic trajectory affinity distributes 12x8=96 episodes evenly across 8 GPUs."""
    import asyncio
    from collections import Counter
    from types import SimpleNamespace

    from rlinf.workers.agent.agent_loop import (
        AgentLoopOutput,
        MultiAgentLoopOutput,
        MultiAgentLoopWorker,
    )

    seen_keys: list[int] = []
    per_traj_keys: list[set[int]] = []

    class _FakeWorker(MultiAgentLoopWorker):
        def __init__(self):
            self.cfg = OmegaConf.create({"rollout": {"replica_affinity": True}})
            self._rollout_num_instances = 8
            self.rollout_request_key = 0
            self.session_server_client = None
            import contextvars

            self._traj_rollout_key = contextvars.ContextVar(
                "test_traj_key", default=None
            )
            self.print_outputs = False
            self.return_logprobs = False
            self.extra_keys_turn = None
            self.extra_keys_traj = None

        async def run_one_query(self, prompt_ids, answer=None):
            keys = set()
            for _ in range(3):
                k = self._traj_rollout_key.get()
                seen_keys.append(k)
                keys.add(k)
                await asyncio.sleep(0)
            per_traj_keys.append(keys)
            return MultiAgentLoopOutput(
                single_turn_outputs=[
                    AgentLoopOutput(prompt_ids=[1], response_ids=[2], reward_score=1.0)
                ]
            )

    class _DummyChan:
        def put(self, item, async_op=True):
            return SimpleNamespace(async_wait=lambda: asyncio.sleep(0))

    async def _run():
        w = _FakeWorker()
        out = _DummyChan()
        await asyncio.gather(
            *(
                w.run_agentloop_rollout_group(
                    [1], "ans", group_size=8, output_channel=out, group_id=gid
                )
                for gid in range(12)
            )
        )

    asyncio.run(_run())
    # Each of 96 trajectories stayed pinned to a single GPU across all 3 turns.
    assert len(per_traj_keys) == 96
    assert all(len(s) == 1 for s in per_traj_keys)
    # Across 8 GPUs, every GPU received exactly 12 trajectories x 3 turns = 36 requests.
    counts = Counter(seen_keys)
    assert dict(counts) == dict.fromkeys(range(8), 36)


def test_rpent_sync_and_async_configs_resolve_and_validate(monkeypatch) -> None:
    """Verify Harness v2 sync, 4B async, and 27B 5-env async configs resolve and validate."""
    from pathlib import Path
    from types import SimpleNamespace

    from hydra import compose, initialize_config_dir

    import rlinf.config as cfg_mod

    cfg_dir = Path(__file__).parents[2] / "examples/agent/rpent/config"
    with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.1"):
        sync_cfg = compose(config_name="train_qwen3.5_4b_rpent_grpo")
        async_4b_cfg = compose(config_name="train_qwen3.5_4b_rpent_async_grpo")
        sync_5env_cfg = compose(config_name="train_qwen3.5_4b_rpent_grpo_5env")
        async_27b_cfg = compose(config_name="train_qwen3.8_27b_rpent_async_grpo_5env")

    assert sync_cfg.runner.synchronous is True
    assert sync_cfg.agentloop.history_compaction.strategy == "dynamic"
    assert sync_cfg.tools.rpent.include_jaw_mm is True
    assert sync_cfg.tools.rpent.image_size is None
    assert len(mixed_simulators(sync_5env_cfg)) == 5

    assert async_4b_cfg.runner.synchronous is False
    assert async_4b_cfg.runner.async_mode is True
    assert async_4b_cfg.rollout.pause_mode == "retract"
    assert async_4b_cfg.rollout_buffer.max_inflight_groups == 6
    assert async_4b_cfg.rollout_buffer.max_buffered_groups == 12
    assert async_4b_cfg.agentloop.max_actor_accept_staleness == 3

    assert async_27b_cfg.runner.synchronous is False
    assert async_27b_cfg.runner.async_mode is True
    assert async_27b_cfg.rollout.tensor_parallel_size == 2
    assert async_27b_cfg.actor.model.tensor_model_parallel_size == 4
    assert async_27b_cfg.actor.model.context_parallel_size == 2
    assert async_27b_cfg.data.rollout_batch_size == 32
    assert async_27b_cfg.algorithm.group_size == 4
    assert async_27b_cfg.rollout_buffer.max_inflight_groups == 36
    assert async_27b_cfg.cluster.component_placement.agentloop == "0-17"
    assert str(async_27b_cfg.agentloop.env_vars.RLINF_TIMEOUT) == "201600"
    assert len(mixed_simulators(async_27b_cfg)) == 5

    # Verify optional RDMA env vars are cleanly omitted when unset and forwarded when set
    import importlib.util

    train_path = Path(__file__).parents[2] / "examples/agent/rpent/train.py"
    spec = importlib.util.spec_from_file_location("rpent_train_entry", train_path)
    assert spec is not None and spec.loader is not None
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    _env_vars = train_mod._env_vars

    monkeypatch.delenv("NCCL_IB_HCA", raising=False)
    resolved_unset = _env_vars(async_27b_cfg.rollout)
    assert resolved_unset is not None
    assert "NCCL_IB_HCA" not in resolved_unset
    monkeypatch.setenv("NCCL_IB_HCA", "mlx5_0,mlx5_1")
    with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.1"):
        async_27b_with_hca = compose(
            config_name="train_qwen3.8_27b_rpent_async_grpo_5env"
        )
    assert _env_vars(async_27b_with_hca.rollout)["NCCL_IB_HCA"] == "mlx5_0,mlx5_1"

    monkeypatch.setattr(cfg_mod, "Cluster", lambda *a, **k: None)
    monkeypatch.setattr(cfg_mod, "version", lambda _name: "0.5.12.post1")

    for cfg_obj, replicas, dp_size in (
        (async_4b_cfg, 4, 2),
        (async_27b_cfg, 24, 6),
    ):
        fake_placement = SimpleNamespace(
            is_collocated=True,
            is_disaggregated=False,
            placement_mode=SimpleNamespace(name="COLLOCATED"),
            rollout_num_model_instances=replicas,
            actor_dp_size=dp_size,
        )
        monkeypatch.setattr(
            cfg_mod,
            "ModelParallelComponentPlacement",
            lambda *a, _p=fake_placement, **k: _p,
        )
        cfg_mod.validate_rpent_async_cfg(cfg_obj)

    # Reject non-retract pause_mode in collocated async mode
    bad_cfg = OmegaConf.create(OmegaConf.to_container(async_4b_cfg, resolve=False))
    bad_cfg.rollout.pause_mode = "in_place"
    with pytest.raises(ValueError, match="pause_mode=retract"):
        cfg_mod.validate_rpent_async_cfg(bad_cfg)


def test_versioned_trajectory_records_and_staleness_metrics():
    """Verify MAMegatronActor._extract_versioned_trajectory_records matches VersionedTrajectoryRecord."""
    from types import SimpleNamespace

    from rlinf.utils.metric_utils import (
        VersionedTrajectoryRecord,
        compute_staleness_metrics,
    )
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor

    actor = MAMegatronActor.__new__(MAMegatronActor)
    result = SimpleNamespace(
        version_start=[0, 1],
        version_end=[1, 1],
        idx_to_traj=[0, 1],
        response_lengths=[4, 3],
        response_mask=[[1, 0, 1, 1], [1, 1, 1]],
        rewards=[1.0, 0.0],
        extra_fields_turn={
            "rollout_weight_versions": [[0, 0, 1, 1], [1, 1, 1]],
        },
    )
    records = actor._extract_versioned_trajectory_records([result])
    assert records is not None
    assert len(records) == 2
    assert isinstance(records[0], VersionedTrajectoryRecord)
    # Valid tokens for traj 0 are indices 0, 2, 3 -> versions [0, 1, 1] -> sum=2
    assert records[0].sampled_token_version_sum == 2
    assert records[1].sampled_token_version_sum == 3

    metrics = compute_staleness_metrics(records, version=2)
    assert metrics["data_staleness/mean"] == 2.0
    assert metrics["data_staleness/max"] == 2
    assert abs(metrics["data_staleness/token_mean"] - (2.0 - 5.0 / 6.0)) < 1e-6
    assert metrics["cross_version_call_ratio"] == 0.5


def test_metaworld_task_pools(tmp_path, monkeypatch):
    prepare_tasks = (
        Path(__file__).resolve().parents[2] / "examples/agent/rpent/prepare_tasks.py"
    )
    write_metaworld_tasks = runpy.run_path(str(prepare_tasks))["write_metaworld_tasks"]

    def initial_scene(name, index, benchmark):
        return SimpleNamespace(close=lambda: None), {
            "task_index": index,
            "rand_vec_sha256": name,
        }

    monkeypatch.setitem(
        sys.modules, "metaworld", SimpleNamespace(MT1=lambda name, seed: name)
    )
    monkeypatch.setitem(
        sys.modules,
        "robots.rl.envs.metaworld",
        SimpleNamespace(initial_scene=initial_scene),
    )
    args = SimpleNamespace(seeds="49", output_dir=tmp_path)
    write_metaworld_tasks(args)
    pools = {}
    for split, count in [("mt50", 50), ("ml45-train", 45), ("ml45-ood", 5)]:
        payload = (tmp_path / f"{split}.jsonl").read_bytes()
        rows = [json.loads(line) for line in payload.splitlines()]
        manifest = json.loads((tmp_path / f"{split}-manifest.json").read_bytes())
        assert len(rows) == count
        assert manifest["files"][split] == {
            "rows": count,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        pools[split] = {r["query_id"] for r in rows}
        for row in rows:
            task = episode_task(row["answer"])
            assert task["environment"] == "metaworld"
            assert (
                task["task_index"]
                == task["init_state_id"]
                == task["seed"]
                == row["initial_state"]["task_index"]
                == 49
            )
    assert pools["ml45-train"].isdisjoint(pools["ml45-ood"])
    assert pools["ml45-train"] | pools["ml45-ood"] == pools["mt50"]
    args.seeds = "50"
    with pytest.raises(ValueError, match="0, 49"):
        write_metaworld_tasks(args)


def test_metaworld_validation_retry():
    from rlinf.agents.rpent.history import EpisodeConversation

    move = {
        "type": "object",
        "properties": {"action": {"const": "move_to"}, "xyz": {"type": "array"}},
        "required": ["action", "xyz"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {"actions": {"type": "array", "items": move}},
    }
    conv = EpisodeConversation(
        system_prompt="SYS",
        user_message="TASK",
        initial_observation_text="{}",
        environment="metaworld",
        tools=[
            {
                "type": "function",
                "function": {"name": "execute_actions", "parameters": schema},
            }
        ],
    )
    action = {"action": "move_to", "xyz": [1, 2, 3], "orientation_rpy": [0, 0, 0]}
    kind, message, parsed, retry = conv.validate_or_retry(
        [{"name": "execute_actions", "arguments": {"actions": [action]}}]
    )
    assert kind == "tool_arguments" and parsed is None and retry
    assert "xyz only" in message


def test_prepare_robolab_tasks(tmp_path) -> None:
    import subprocess

    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "source_revision": "bb" * 20,
                "source_metadata_sha256": "aa" * 32,
                "task_count": 2,
                "tasks": {
                    "StackBlocksTask": {"instruction": "Stack the blocks."},
                    "BananaInBowlTask": {"instruction": "Put the banana in the bowl."},
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "episodes"
    command = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[2]
            / "examples/agent/rpent/prepare_tasks.py"
        ),
        "--robolab-catalog",
        str(catalog),
        "--output-dir",
        str(out),
        "--split",
        "eval",
    ]
    missing_seeds = subprocess.run(command, capture_output=True, text=True, check=False)
    assert missing_seeds.returncode == 2
    assert "RoboLab requires explicit --seeds" in missing_seeds.stderr
    subprocess.run([*command, "--seeds", "2,0-2"], check=True)
    payload = (out / "eval.jsonl").read_bytes()
    rows = [json.loads(line) for line in payload.splitlines()]
    assert [row["query_id"] for row in rows] == [
        "robolab-BananaInBowlTask-seed0",
        "robolab-BananaInBowlTask-seed1",
        "robolab-BananaInBowlTask-seed2",
        "robolab-StackBlocksTask-seed0",
        "robolab-StackBlocksTask-seed1",
        "robolab-StackBlocksTask-seed2",
    ]
    assert rows[0]["prompt"] == "Put the banana in the bowl."
    assert episode_task(rows[-1]["answer"]) == {
        "environment": "robolab",
        "task_name": "StackBlocksTask",
        "seed": 2,
    }
    manifest = json.loads((out / "eval-manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_revision"] == "bb" * 20
    assert (
        manifest["catalog_sha256"] == hashlib.sha256(catalog.read_bytes()).hexdigest()
    )
    assert manifest["files"]["eval"] == {
        "rows": 6,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


@pytest.fixture
def mixed_robolab_cfg(monkeypatch):
    from hydra import compose, initialize_config_dir

    monkeypatch.setenv("ROBOLAB_POOL_URL", "http://robolab.invalid")
    monkeypatch.setenv("ROBOTWIN_POOL_URL", "http://robotwin.invalid")
    config_dir = Path(__file__).resolve().parents[2] / "examples/agent/rpent/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        return compose(config_name="train_qwen3.5_4b_rpent_robolab_smoke")


def test_mixed_robolab_smoke_capacity(mixed_robolab_cfg) -> None:
    from rlinf.runners.rpent_runner import validate_sim_pool_capacity

    simulators = mixed_simulators(mixed_robolab_cfg)
    assert {name: spec.endpoint for name, spec in simulators.items()} == {
        "robolab": "http://robolab.invalid",
        "robotwin": "http://robotwin.invalid",
    }
    assert all(
        spec.rollout_groups == spec.sampling_groups == 1 for spec in simulators.values()
    )
    kwargs = {
        "simulators": simulators,
        "default_endpoint": None,
        "sampling_groups": mixed_robolab_cfg.data.sampling_batch_size,
        "group_size": mixed_robolab_cfg.algorithm.group_size,
        "num_agent_workers": 4,
        "episodes_per_worker": None,
    }
    statuses = validate_sim_pool_capacity(
        **kwargs,
        fetch_health=lambda _: {"workers_total": 2, "workers_idle": 2},
    )
    assert {name: status["required"] for name, status in statuses.items()} == {
        "robolab": 2,
        "robotwin": 2,
    }
    with pytest.raises(RuntimeError, match=r"'robolab'.*workers_total=1"):
        validate_sim_pool_capacity(
            **kwargs,
            fetch_health=lambda endpoint: {
                "workers_total": 1 if endpoint == "http://robolab.invalid" else 2,
                "workers_idle": 1,
            },
        )


@pytest.mark.parametrize("outcome", ["success", "timeout", "infrastructure"])
def test_robolab_episode_uses_native_outcome(mixed_robolab_cfg, outcome) -> None:
    import asyncio

    import httpx

    from rlinf.agents.rpent.agent_loop import RPentAgentLoopWorker, RPentEpisodeError

    task = {"environment": "robolab", "task_name": "BananaInBowlTask", "seed": 7}
    arguments = {"actions": [{"action": "set_gripper", "opening": 0}]}
    call = {"name": "execute_actions", "arguments": arguments, "id": "call_0_0"}
    tool = {
        "type": "function",
        "function": {
            "name": "execute_actions",
            "parameters": {
                "type": "object",
                "properties": {"actions": {"type": "array"}},
                "required": ["actions"],
            },
        },
    }
    requests = []

    def serve(request):
        assert request.url.host == "robolab.invalid"
        requests.append(request.url.path)
        if request.url.path == "/sessions/open":
            assert json.loads(request.content)["task"] == task
            return httpx.Response(
                200,
                json={
                    "session_id": "robolab-episode",
                    "system_prompt": "Control the robot.",
                    "user_message": "Put the banana in the bowl.",
                    "initial_observation": "The gripper is open.",
                    "tools": [tool],
                    "environment": "robolab",
                    "done": False,
                    "success": False,
                    "truncated": False,
                },
            )
        if request.url.path == "/sessions/robolab-episode/execute":
            assert json.loads(request.content) == {
                "name": "execute_actions",
                "arguments": arguments,
            }
            return httpx.Response(
                200,
                json={
                    "text": "The episode ended.",
                    "done": True,
                    "success": outcome == "success",
                    "truncated": outcome == "timeout",
                    "infrastructure_error": "simulator disconnected"
                    if outcome == "infrastructure"
                    else None,
                },
            )
        assert request.url.path == "/sessions/robolab-episode/close"
        return httpx.Response(200, json={"closed": True})

    async def sample_planner_turn(**_):
        return AgentLoopOutput(prompt_ids=[1], response_ids=[2]), [call], {}, "stop", 0

    async def run():
        worker = object.__new__(RPentAgentLoopWorker)
        worker.cfg = mixed_robolab_cfg
        worker.cfg.agentloop.max_infra_retries = 0
        worker.cfg.agentloop.merge_prefix_turns = False
        worker._simulators = mixed_simulators(worker.cfg)
        worker._episode_sem = None
        worker._compaction = None
        worker._thinking_enabled = True
        worker._sample_turn = sample_planner_turn
        worker.agentic_tokenizer = SimpleNamespace(
            tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 0)
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            worker._http = client
            if outcome == "infrastructure":
                with pytest.raises(
                    RPentEpisodeError, match="simulator disconnected"
                ) as error:
                    await worker.run_one_query([1], [json.dumps(task)])
                assert error.value.kind == "tool_transport"
            else:
                output = await worker.run_one_query([1], [json.dumps(task)])
                assert output.single_turn_outputs[0].reward_score == float(
                    outcome == "success"
                )
                assert output.extra_fields["truncated"] == (outcome == "timeout")
        assert requests == [
            "/sessions/open",
            "/sessions/robolab-episode/execute",
            "/sessions/robolab-episode/close",
        ]

    asyncio.run(run())


def test_harness_v2_rlinf_rpent_parity_and_chain_switch() -> None:
    """Verify RLinf history.py matches RPent conversation.py on opening images and dynamic compaction."""
    import io

    from PIL import Image

    from rlinf.agents.rpent.history import (
        AppendOnlyHistory,
        CompactionConfig,
        EpisodeConversation,
    )

    def _png(rgb: tuple[int, int, int]) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), rgb).save(buf, format="PNG")
        return buf.getvalue()

    class _FakeInnerTok:
        def convert_tokens_to_ids(self, tok: str) -> int:
            return 999 if tok == "<|image_pad|>" else 1

        def __call__(self, text: str, add_special_tokens: bool = False):
            return {
                "input_ids": [
                    999 if "<|image_pad|>" in part else (abs(hash(part)) % 100 + 10)
                    for part in text.split()
                ]
            }

    class _FakeAgenticTok:
        def __init__(self):
            self.tokenizer = _FakeInnerTok()

        def apply_chat_template(
            self, messages, tools=None, tokenize=False, add_generation_prompt=True
        ):
            chunks = []
            for m in messages:
                if m.get("reasoning_content"):
                    chunks.append(f"<think>{m['reasoning_content']}</think>")
                c = m.get("content", "")
                if isinstance(c, list):
                    for part in c:
                        chunks.append(
                            "<|image_pad|>"
                            if part.get("type") == "image_url"
                            else part.get("text", "")
                        )
                else:
                    chunks.append(str(c))
            return " ".join(chunks)

        def render_incremental(self, delta_msgs):
            return self.tokenizer(self.apply_chat_template(delta_msgs))["input_ids"]

        def join_increment(self, prefix_ids, increment):
            return [*prefix_ids, *increment]

    png_open_0 = _png((10, 20, 30))
    png_open_1 = _png((40, 50, 60))
    step_imgs = [_png((100 + i * 20, 10, 10)) for i in range(5)]

    cfg = CompactionConfig.from_cfg(
        {
            "strategy": "dynamic",
            "max_messages": None,
            "max_tokens": 200,
            "keep_messages": 4,
            "keep_recent_images": 2,
            "keep_recent_thinking": 2,
            "preserve_opening_images": True,
        }
    )
    conv = EpisodeConversation(
        system_prompt="SYS",
        user_message="TASK",
        initial_observation_text='{"step": 0}',
        tools=[],
        environment="libero",
        compaction=cfg,
        initial_images=[png_open_0, png_open_1],
        initial_image_cameras=["agentview", "wrist"],
    )
    hist = AppendOnlyHistory(_FakeAgenticTok())

    # Turn 0: opening message carries 2 step=0 images with prefix_length=0
    r0 = hist.render_turn(conv.wire_messages(), conv.tools, conv.images())
    assert r0.prefix_length == 0
    assert r0.images == [png_open_0, png_open_1]
    assert len(r0.image_spans) == 2
    assert hist.chain_id == 1

    # Simulate accept_capture expanding <|image_pad|> tokens on Turn 0
    expanded_ids = list(r0.input_ids)
    captured = {
        "prompt_token_ids": expanded_ids,
        "image_spans": r0.image_spans,
    }
    assert hist.accept_capture(captured) == expanded_ids
    hist.image_history = {"length": len(expanded_ids), "images": ["img0", "img1"]}

    # Run 5 turns adding images & thinking without compacting yet
    last_wire = conv.wire_messages()
    last_ids = expanded_ids
    for i, img in enumerate(step_imgs):
        asst = {
            "role": "assistant",
            "content": None,
            "reasoning_content": f"think {i} " + ("z" * 120),
            "tool_calls": [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "execute_actions", "arguments": "{}"},
                }
            ],
        }
        hist.record(last_wire, conv.tools, last_ids, [50 + i], asst)
        conv.append_assistant(asst)
        conv.append_tool_result(
            tool_name="execute_actions",
            tool_call_id=f"c{i}",
            text=f'{{"step": {i + 1}}}',
            images=[img],
        )
        last_wire = conv.wire_messages()
        rendered = hist.render_turn(last_wire, conv.tools, conv.images())
        assert hist.chain_id == 1
        assert len(rendered.image_spans) == len(rendered.images) == 2 + i + 1
        last_ids = rendered.input_ids

    # Snapshot pre-compaction wire messages stored in AppendOnlyHistory
    saved_history_msgs = list(hist._messages)  # type: ignore[arg-type]
    saved_len = len(saved_history_msgs)

    # Trigger dynamic compaction
    assert conv.maybe_compact(current_tokens=1000) is True
    # Opening images (2) + last 2 intermediate images = 4 images total
    assert conv.images() == [png_open_0, png_open_1, step_imgs[-2], step_imgs[-1]]
    # AppendOnlyHistory's saved prefix must not have been mutated
    assert len(hist._messages) == saved_len  # type: ignore[arg-type]
    # Next render_turn must detect prefix break, increment chain_id, and reset image_history
    assert hist._continues(conv.wire_messages(), conv.tools) is False
    r_after = hist.render_turn(conv.wire_messages(), conv.tools, conv.images())
    assert hist.chain_id == 2
    assert hist.image_history is None
    assert r_after.prefix_length == 0
    assert len(r_after.image_spans) == len(r_after.images) == 4

    # Cross-repo parity check against RPent EpisodeConversation if RPent is importable
    try:
        from robots.rl.conversation import (
            CompactionConfig as RPentCompactionConfig,
        )
        from robots.rl.conversation import (
            EpisodeConversation as RPentEpisodeConversation,
        )
    except ImportError:
        return
    rpent_conv = RPentEpisodeConversation(
        system_prompt="SYS",
        user_message="TASK",
        initial_observation_text='{"step": 0}',
        tools=[],
        environment="libero",
        compaction=RPentCompactionConfig(
            strategy="dynamic",
            max_messages=None,
            max_tokens=200,
            keep_messages=4,
            keep_recent_images=2,
            keep_recent_thinking=2,
            preserve_opening_images=True,
        ),
        initial_images=[png_open_0, png_open_1],
        initial_image_cameras=["agentview", "wrist"],
    )
    for i, img in enumerate(step_imgs):
        rpent_conv.append_assistant(
            content=None,
            reasoning_content=f"think {i} " + ("z" * 120),
            tool_calls=[
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "execute_actions", "arguments": "{}"},
                }
            ],
        )
        rpent_conv.append_tool_result(
            tool_name="execute_actions",
            tool_call_id=f"c{i}",
            text=f'{{"step": {i + 1}}}',
            images=[img],
        )
    assert rpent_conv.maybe_compact(current_tokens=1000) is True
    assert rpent_conv.wire_messages() == conv.wire_messages()
    assert rpent_conv.images() == conv.images()


def test_harness_v2_5env_mixed_simulators_and_binary_open() -> None:
    """Verify 5-env joint RL quota validation and binary /sessions/open unpacking."""
    import asyncio
    import struct

    import httpx

    from rlinf.agents.rpent.agent_loop import (
        _BINARY_EXECUTE_MEDIA_TYPE,
        RPentAgentLoopWorker,
    )

    cfg = OmegaConf.create(
        {
            "algorithm": {"group_size": 8},
            "data": {"rollout_batch_size": 5, "sampling_batch_size": 10},
            "env": {
                "rpent": {
                    env_name: {
                        "endpoint": f"http://{env_name}.invalid:28560",
                        "train_data_paths": f"/tmp/{env_name}.jsonl",
                        "rollout_episodes": 8,
                        "sampling_episodes": 16,
                    }
                    for env_name in (
                        "libero",
                        "robotwin",
                        "robodojo",
                        "robolab",
                        "metaworld",
                    )
                }
            },
            "tools": {
                "rpent": {
                    "max_episode_steps": 10000,
                    "max_actions_per_call": 3,
                    "image_size": None,
                    "include_jaw_mm": True,
                    "record_artifacts": False,
                }
            },
        }
    )
    sims = mixed_simulators(cfg)
    assert set(sims) == {"libero", "robotwin", "robodojo", "robolab", "metaworld"}
    assert all(s.rollout_groups == 1 and s.sampling_groups == 2 for s in sims.values())

    png_bytes = b"\x89PNG\r\n\x1a\nstep0"
    seen_payload: dict = {}

    def serve(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("accept") == _BINARY_EXECUTE_MEDIA_TYPE
        seen_payload.update(json.loads(request.content))
        meta = {
            "session_id": "sid-bin-0",
            "environment": "metaworld",
            "system_prompt": "SYS",
            "user_message": "USR",
            "initial_observation": "{}",
            "image_cameras": ["corner2"],
            "tools": [],
            "done": False,
            "success": False,
            "truncated": False,
        }
        meta_b = json.dumps(meta).encode("utf-8")
        body = (
            struct.pack("!I", len(meta_b))
            + meta_b
            + struct.pack("!I", 1)
            + struct.pack("!I", len(png_bytes))
            + png_bytes
        )
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": _BINARY_EXECUTE_MEDIA_TYPE},
        )

    async def _run() -> None:
        worker = object.__new__(RPentAgentLoopWorker)
        worker.cfg = cfg
        worker._thinking_enabled = True
        worker._generation_cohort_id = "coh-1"
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            opened, imgs = await worker._open_session(
                client,
                "http://metaworld.invalid:28560",
                {"environment": "metaworld", "task_name": "reach-v3", "seed": 0},
            )
            assert opened["session_id"] == "sid-bin-0"
            assert imgs == [png_bytes]
            assert seen_payload["image_size"] is None
            assert seen_payload["include_jaw_mm"] is True

    asyncio.run(_run())


def test_uneven_dp_pad_rows_balance_and_loss_normalization(monkeypatch) -> None:
    """Verify uneven-DP group balancing, CP>1 dummy row padding, and unbiased scaling."""
    import torch

    from rlinf.algorithms.loss_scales import restore_source_dp_normalization
    from rlinf.data.schema.actor_batch import pad_actor_batch_rows
    from rlinf.utils.distributed import balance_rollout_groups

    # 1. pad_actor_batch_rows with CP>1 min_row_len = tp * cp * 2 = 16
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: tensor.fill_(4),
    )
    batch = {
        "input_ids": [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
        "attention_mask": [
            torch.tensor([True, True, True]),
            torch.tensor([True, True]),
        ],
        "response_mask": [
            torch.tensor([False, True, True]),
            torch.tensor([False, True]),
        ],
        "advantages": [
            torch.tensor([0.0, 1.0, 1.0]),
            torch.tensor([0.0, -1.0]),
        ],
        "loss_scales": [
            torch.tensor([0.0, 0.5, 0.5]),
            torch.tensor([0.0, 1.0]),
        ],
        "idx_to_traj": [0, 1],
        "multi_modal_inputs": [{"pixel_values": torch.zeros(1)}, None],
        "rewards": torch.tensor([1.0, 0.0]),
    }
    padded = pad_actor_batch_rows(
        batch, divisor=2, dp_group=None, device="cpu", min_row_len=16
    )
    assert len(padded["input_ids"]) == 4
    assert len(padded["multi_modal_inputs"]) == 4
    assert padded["multi_modal_inputs"][2:] == [None, None]
    assert padded["idx_to_traj"][2:] == [0, 0]
    assert padded["rewards"].tolist() == [1.0, 0.0, 0.0, 0.0]
    for dummy_idx in (2, 3):
        assert padded["input_ids"][dummy_idx].shape == (16,)
        assert padded["attention_mask"][dummy_idx].shape == (16,)
        assert padded["attention_mask"][dummy_idx].all()
        assert not padded["response_mask"][dummy_idx].any()
        assert padded["advantages"][dummy_idx].abs().sum().item() == 0.0
        assert padded["loss_scales"][dummy_idx].abs().sum().item() == 0.0

    # 2. balance_rollout_groups with 16 groups across 6 DP ranks (uneven: 3,3,3,3,2,2)
    world_size = 6
    per_rank_costs = [
        [1000, 900, 100],
        [800, 700, 200],
        [600, 500, 300],
        [450, 400, 350],
        [250, 150],
        [120, 80],
    ]
    # Simulate all 6 ranks in-process with a shared in-memory all_to_all buffer
    rank_results = [
        [
            SimpleNamespace(input_ids=[list(range(c))], tag=(r, i))
            for i, c in enumerate(costs)
        ]
        for r, costs in enumerate(per_rank_costs)
    ]

    def _run_balance_all_ranks():
        # Run a two-phase simulation: capture sends from all ranks, then deliver receives
        monkeypatch.setattr(
            torch.distributed, "get_world_size", lambda g=None: world_size
        )
        monkeypatch.setattr(torch.distributed, "get_backend", lambda g=None: "gloo")

        def _fake_all_gather_obj(out, obj, group=None):
            for idx, rank_costs in enumerate(per_rank_costs):
                out[idx] = rank_costs

        monkeypatch.setattr(
            torch.distributed, "all_gather_object", _fake_all_gather_obj
        )

        # Phase A: capture outgoing sizes & byte payloads from each rank
        send_chunks: list[list[bytes]] = [[b""] * world_size for _ in range(world_size)]
        for r in range(world_size):
            monkeypatch.setattr(torch.distributed, "get_rank", lambda g=None, _r=r: _r)
            sizes_seen: list[list[int]] = []

            def _capture_a2a(
                output,
                input,
                output_split_sizes=None,
                input_split_sizes=None,
                group=None,
                _r=r,
            ):
                if input_split_sizes is None:
                    sizes_seen.append(input.tolist())
                    output.zero_()
                else:
                    raw = input.numpy().tobytes()
                    off = 0
                    for dst, sz in enumerate(input_split_sizes):
                        send_chunks[_r][dst] = raw[off : off + sz]
                        off += sz

            monkeypatch.setattr(torch.distributed, "all_to_all_single", _capture_a2a)
            try:
                balance_rollout_groups(rank_results[r], dp_group=None)
            except KeyError:
                pass  # Expected in capture phase when recv is empty

        # Phase B: deliver real all-to-all payloads and collect final balanced buckets
        balanced_buckets = []
        for r in range(world_size):
            monkeypatch.setattr(torch.distributed, "get_rank", lambda g=None, _r=r: _r)

            def _deliver_a2a(
                output,
                input,
                output_split_sizes=None,
                input_split_sizes=None,
                group=None,
                _r=r,
            ):
                if input_split_sizes is None:
                    for src in range(world_size):
                        output[src] = len(send_chunks[src][_r])
                else:
                    joined = b"".join(send_chunks[src][_r] for src in range(world_size))
                    if joined:
                        output.copy_(
                            torch.frombuffer(bytearray(joined), dtype=torch.uint8)
                        )

            monkeypatch.setattr(torch.distributed, "all_to_all_single", _deliver_a2a)
            balanced_buckets.append(
                balance_rollout_groups(rank_results[r], dp_group=None)
            )
        return balanced_buckets

    balanced = _run_balance_all_ranks()
    assert len(balanced) == world_size
    assert all(len(bucket) >= 1 for bucket in balanced)
    all_tags = [item.tag for bucket in balanced for item in bucket]
    assert len(all_tags) == 16
    assert len(set(all_tags)) == 16

    # 3. restore_source_dp_normalization with uneven_dp_groups=True normalizes uniformly
    # Suppose global total sequences = 64 across world_size = 6, local rank holds 8 sequences.
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda g=None: 6)
    global_counts = torch.tensor([12, 12, 12, 12, 8, 8], dtype=torch.long)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, op=None, group=None: tensor.copy_(global_counts),
    )
    norm_batch = {
        "extra:source_dp": torch.tensor([0, 0, 1, 2, 3, 4, 5, 5], dtype=torch.long),
        "advantages": [torch.tensor([1.0]) for _ in range(8)],
    }
    ctx = {"folding_scale": ["group_level"], "uneven_dp_groups": True}
    restore_source_dp_normalization(norm_batch, ctx, dp_group=None, device="cpu")
    expected_ratio = 64.0 / (6 * 8)
    for adv in norm_batch["advantages"]:
        assert adv.item() == pytest.approx(expected_ratio)


def test_aggregate_agent_metrics_handles_sparse_per_env_keys() -> None:
    """Verify multi-env async agent metrics tolerate keys absent on some workers."""
    from rlinf.utils.metric_utils import aggregate_agent_metrics

    agg = aggregate_agent_metrics(
        [
            {
                "__mean__/rpent/success_rate": (1, 2),
                "__mean__/rpent/libero/success_rate": (1, 2),
                "__sum__/rpent/episodes": 2,
            },
            {
                "__mean__/rpent/success_rate": (2, 2),
                "__mean__/rpent/robolab/success_rate": (2, 2),
                "__sum__/rpent/episodes": 2,
                "__max__/rpent/max_turns": 6,
            },
        ]
    )
    assert agg["rpent/success_rate"] == pytest.approx(0.75)
    assert agg["rpent/libero/success_rate"] == pytest.approx(0.5)
    assert agg["rpent/robolab/success_rate"] == pytest.approx(1.0)
    assert agg["rpent/episodes"] == pytest.approx(4.0)
    assert agg["rpent/max_turns"] == pytest.approx(6.0)


def test_rpent_async_infra_failure_discards_group_instead_of_crashing() -> None:
    """In async mode, exhausting infra retries raises GroupDiscarded; sync mode still raises."""
    import asyncio

    from rlinf.agents.rpent.agent_loop import RPentAgentLoopWorker, RPentEpisodeError
    from rlinf.workers.agent.agent_loop import GroupDiscarded

    task_json = json.dumps(
        {"environment": "robolab", "task_name": "BrokenTask", "seed": 0}
    )

    def _make_worker(synchronous: bool):
        worker = RPentAgentLoopWorker.__new__(RPentAgentLoopWorker)
        worker.cfg = OmegaConf.create(
            {
                "runner": {"synchronous": synchronous, "async_mode": not synchronous},
                "agentloop": {"max_infra_retries": 1},
            }
        )
        warnings: list[str] = []
        worker.log_warning = warnings.append

        async def _always_fail(_task):
            raise RPentEpisodeError(
                "HTTP 500 AttributeError: no bowl", kind="env_setup"
            )

        worker._run_episode = _always_fail
        return worker, warnings

    async_worker, async_warnings = _make_worker(synchronous=False)
    with pytest.raises(GroupDiscarded) as exc_info:
        asyncio.run(async_worker.run_one_query([], [task_json]))
    assert exc_info.value.reason == "rpent_env_setup"
    assert any("Discarding async RPent group" in w for w in async_warnings)

    sync_worker, _ = _make_worker(synchronous=True)
    with pytest.raises(RPentEpisodeError, match="no bowl"):
        asyncio.run(sync_worker.run_one_query([], [task_json]))
