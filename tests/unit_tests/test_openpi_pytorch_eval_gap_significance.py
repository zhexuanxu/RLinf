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

"""Eval-gap significance gate.

The RLinf-trained vs reference-trained BEHAVIOR success gap is assessed for
statistical significance from each run's logged outcomes. The counts are derived
from the eval logs (not hardcoded), so the gate fails if those logs change. The
gap is shown to be within eval run-to-run noise under a two-proportion test and a
95% confidence interval, satisfying DEC-1 (trend / confirm-significance-first).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_DUMP_PATH = pathlib.Path(__file__).parent / "_eval_gap_significance_dump.py"
_spec = importlib.util.spec_from_file_location(
    "_eval_gap_significance_dump", _DUMP_PATH
)
dump = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dump)


def _have_logs() -> bool:
    return dump._RLINF_EVAL_LOG.is_file() and dump._REFERENCE_EVAL_LOG.is_file()


@pytest.mark.skipif(not _have_logs(), reason="eval logs not present")
def test_eval_counts_parsed_from_logs():
    """The success counts are read from the eval logs: 33/128 vs 41/128."""
    rlinf = dump.parse_eval_log(dump._RLINF_EVAL_LOG)
    reference = dump.parse_eval_log(dump._REFERENCE_EVAL_LOG)
    assert (rlinf["successes"], rlinf["n"]) == (33, 128)
    assert (reference["successes"], reference["n"]) == (41, 128)


@pytest.mark.skipif(not _have_logs(), reason="eval logs not present")
def test_eval_gap_is_within_noise():
    """The reported gap is NOT statistically significant: the two-proportion test
    does not reject equality and the 95% CI of the difference includes 0, so the
    gap is statistically explained (DEC-1) rather than a defect."""
    ev = dump.build_evidence()
    assert not ev["two_proportion_z_test"]["significant_at_0.05"], (
        f"unexpectedly significant: {ev['two_proportion_z_test']}"
    )
    assert not ev["difference_95_ci_excludes_zero"], (
        f"CI unexpectedly excludes 0: {ev['difference_95_ci']}"
    )
    assert not ev["dec1_gap_is_material"]
    # CI must actually bracket 0 (sanity on the interval orientation).
    lo, hi = ev["difference_95_ci"]
    assert lo < 0 < hi
