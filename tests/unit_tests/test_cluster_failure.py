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

"""Worker-failure cleanup must work without a Ray Dashboard."""

import importlib
import logging
import signal
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def cleanup(monkeypatch):
    module = importlib.import_module("rlinf.scheduler.cluster.cluster")
    monkeypatch.setattr(module.Cluster, "NAMESPACE", "this-run")
    looked_up = []
    killed = []
    shutdown = []

    def get_actor(name, *, namespace):
        looked_up.append((name, namespace))
        return name

    monkeypatch.setattr(module.ray, "get_actor", get_actor)
    monkeypatch.setattr(
        module.ray,
        "kill",
        lambda actor, *, no_restart: killed.append((actor, no_restart)),
    )
    monkeypatch.setattr(module.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        module.ray,
        "shutdown",
        lambda **kwargs: shutdown.append(kwargs),
    )
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    state = SimpleNamespace(
        _distributed_log_collector=None,
        _logger=logging.getLogger("test.cluster.worker_failure"),
    )
    return SimpleNamespace(
        module=module,
        state=state,
        looked_up=looked_up,
        killed=killed,
        shutdown=shutdown,
    )


def _fail_worker(cleanup, capsys):
    print("original worker error: missing scene asset", file=sys.stderr)
    with pytest.raises(SystemExit) as result:
        cleanup.module.Cluster._handle_worker_failure(
            cleanup.state, signal.SIGUSR1, None
        )
    assert result.value.code == -1
    output = capsys.readouterr()
    assert "original worker error: missing scene asset" in output.err
    assert "failure upon worker execution" in output.out
    assert cleanup.shutdown == [{"_exiting_interpreter": True}]


def test_state_api_cleanup_remains_namespace_scoped(cleanup, monkeypatch, capsys):
    requests = []

    def list_actors(**kwargs):
        requests.append(kwargs)
        return [
            SimpleNamespace(name="worker", ray_namespace="this-run"),
            SimpleNamespace(name="foreign", ray_namespace="another-run"),
            SimpleNamespace(name="", ray_namespace="this-run"),
        ]

    monkeypatch.setattr(cleanup.module, "list_actors", list_actors)

    def gcs_must_not_be_used(**kwargs):
        pytest.fail("A working State API must keep its normal cleanup path")

    monkeypatch.setattr(
        cleanup.module.ray.util, "list_named_actors", gcs_must_not_be_used
    )
    _fail_worker(cleanup, capsys)
    assert requests == [
        {"filters": [("STATE", "=", "ALIVE"), ("RAY_NAMESPACE", "=", "this-run")]}
    ]
    assert cleanup.looked_up == [("worker", "this-run")]
    assert cleanup.killed == [("worker", True)]


@pytest.mark.parametrize("gcs_unavailable", [False, True])
def test_headless_fallback_preserves_worker_failure_and_namespace(
    cleanup, monkeypatch, capsys, caplog, gcs_unavailable
):
    def no_dashboard(**kwargs):
        raise RuntimeError("Dashboard port 8265 unavailable")

    requests = []

    def named_actors(*, all_namespaces):
        requests.append(all_namespaces)
        if gcs_unavailable:
            raise RuntimeError("GCS discovery unavailable too")
        return [
            {"name": "worker", "namespace": "this-run"},
            {"name": "worker", "namespace": "another-run"},
            {"name": "foreign", "namespace": "another-run"},
            {"name": "", "namespace": "this-run"},
        ]

    monkeypatch.setattr(cleanup.module, "list_actors", no_dashboard)
    monkeypatch.setattr(cleanup.module.ray.util, "list_named_actors", named_actors)
    _fail_worker(cleanup, capsys)
    assert requests == [True]
    assert cleanup.looked_up == ([] if gcs_unavailable else [("worker", "this-run")])
    assert cleanup.killed == ([] if gcs_unavailable else [("worker", True)])
    assert "listing named actors through GCS" in caplog.text


def test_disappearing_actor_does_not_skip_remaining_cleanup(
    cleanup, monkeypatch, capsys
):
    monkeypatch.setattr(
        cleanup.module,
        "list_actors",
        lambda **kwargs: [
            SimpleNamespace(name="gone", ray_namespace="this-run"),
            SimpleNamespace(name="worker", ray_namespace="this-run"),
        ],
    )

    def get_actor(name, *, namespace):
        cleanup.looked_up.append((name, namespace))
        if name == "gone":
            raise ValueError("actor already exited")
        return name

    monkeypatch.setattr(cleanup.module.ray, "get_actor", get_actor)
    _fail_worker(cleanup, capsys)
    assert cleanup.looked_up == [("gone", "this-run"), ("worker", "this-run")]
    assert cleanup.killed == [("worker", True)]
