"""CI workflow guard (roadmap P0-2): every check blocks.

The `lint` job was advisory (`continue-on-error: true`) while eight `N999` findings waited for
decision D-9, and a red step in a green job is easy to stop looking at. Those findings are gone
(the modules were renamed, ADR 0023), so the job is blocking. This test fails if an advisory flag
comes back on any job, and if one of the three jobs is dropped.

The workflow is read as plain text, so the test needs neither network nor a YAML parser.
"""

import re
from pathlib import Path

_WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yaml"
_JOBS = ("lint", "types", "tests")


def _activeLines() -> list[str]:
    """The workflow lines that are not comments."""
    lines = _WORKFLOW.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if not line.lstrip().startswith("#")]


def test_every_job_is_present() -> None:
    headers = {
        match.group(1) for line in _activeLines() if (match := re.match(r"^  ([a-z]+):\s*$", line))
    }
    assert set(_JOBS) <= headers


def test_no_job_is_advisory() -> None:
    advisory = [line for line in _activeLines() if "continue-on-error" in line]
    assert advisory == []
