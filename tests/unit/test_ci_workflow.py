"""CI workflow guard (roadmap P0-2 and P5-9): every check blocks, and the weekly run is there.

The `lint` job was advisory (`continue-on-error: true`) while eight `N999` findings waited for
decision D-9, and a red step in a green job is easy to stop looking at. Those findings are gone
(the modules were renamed, ADR 0023), so the job is blocking. This test fails if an advisory flag
comes back on any job, and if one of the three jobs is dropped.

Roadmap P5-9 (ADR 0050) adds four things that a later edit could undo without a red run: a weekly
`schedule:` (the only thing that catches a new release of ruff, mypy, numpy or torch between two
pushes), a coverage floor that follows the measured value (the comment that quotes it went stale
once, at 96.25% against 98.51%), the `--durations=15` report in the `tests` job, and the branch
coverage job, which is a report and must stay one: no floor, and not on a push.

The workflow is read as plain text, so the test needs neither network nor a YAML parser.
"""

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yaml"
_JOBS = ("lint", "types", "tests")
_BRANCH_JOB = "branch-coverage"
_JOB_HEADER = re.compile(r"^  ([a-z][a-z-]*):\s*$")
_MINIMUM_FLOOR = 97


def _rawLines() -> list[str]:
    return _WORKFLOW.read_text(encoding="utf-8").splitlines()


def _activeLines() -> list[str]:
    """The workflow lines that are not comments."""
    return [line for line in _rawLines() if not line.lstrip().startswith("#")]


def _jobHeaders() -> set[str]:
    """The names of the jobs, from the two-space-indented keys under `jobs:`."""
    names: set[str] = set()
    in_jobs = False
    for line in _activeLines():
        if line.startswith("jobs:"):
            in_jobs = True
        elif in_jobs and (match := _JOB_HEADER.match(line)):
            names.add(match.group(1))
    return names


def _jobBlock(name: str) -> list[str]:
    """The active lines of one job, from its header to the next job (or the end of the file)."""
    block: list[str] = []
    inside = False
    in_jobs = False
    for line in _activeLines():
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        header = _JOB_HEADER.match(line)
        if header:
            inside = header.group(1) == name
            continue
        if inside:
            block.append(line)
    return block


def _floor() -> int:
    for line in _activeLines():
        if match := re.match(r'^\s+COVERAGE_FLOOR:\s*"(\d+)"\s*$', line):
            return int(match.group(1))
    raise AssertionError("COVERAGE_FLOOR is not set in the env block of ci.yaml")


def _floorComment() -> str:
    """The comment lines directly above `COVERAGE_FLOOR`, as one text."""
    lines = _rawLines()
    index = next(i for i, line in enumerate(lines) if re.match(r"^\s+COVERAGE_FLOOR:", line))
    comment: list[str] = []
    for line in reversed(lines[:index]):
        if not line.lstrip().startswith("#"):
            break
        comment.append(line.lstrip().lstrip("#").strip())
    return " ".join(reversed(comment))


def test_every_job_is_present() -> None:
    assert set(_JOBS) <= _jobHeaders()


def test_no_job_is_advisory() -> None:
    advisory = [line for line in _activeLines() if "continue-on-error" in line]
    assert advisory == []


class TestTheWeeklyRun:
    def _crons(self) -> list[str]:
        return [
            match.group(1)
            for line in _activeLines()
            if (match := re.match(r'^\s+- cron:\s*"([^"]+)"\s*$', line))
        ]

    def test_the_workflow_has_a_schedule_trigger(self) -> None:
        assert any(re.match(r"^  schedule:\s*$", line) for line in _activeLines())

    def test_it_runs_once_a_week_and_no_more_often(self) -> None:
        crons = self._crons()
        assert len(crons) == 1
        fields = crons[0].split()
        assert len(fields) == 5
        minute, hour, day_of_month, month, weekday = fields
        assert minute.isdigit() and hour.isdigit()
        assert day_of_month == "*" and month == "*"
        assert weekday.isdigit() and 0 <= int(weekday) <= 7

    def test_it_runs_the_same_blocking_jobs_as_a_push(self) -> None:
        for name in _JOBS:
            conditions = [line for line in _jobBlock(name) if re.match(r"^    if:", line)]
            assert conditions == [], f"job {name} is conditional: a push or a schedule may skip it"

    def test_the_push_and_pull_request_triggers_remain(self) -> None:
        active = _activeLines()
        assert any(re.match(r"^  push:\s*$", line) for line in active)
        assert any(re.match(r"^  pull_request:\s*$", line) for line in active)


class TestTheCoverageFloor:
    def test_it_is_not_below_the_value_of_roadmap_p5_9(self) -> None:
        assert _floor() >= _MINIMUM_FLOOR

    def test_the_pytest_step_reads_it_from_the_env_block(self) -> None:
        assert any('--cov-fail-under="$COVERAGE_FLOOR"' in line for line in _jobBlock("tests"))

    def test_its_comment_quotes_a_value_one_to_two_points_above_it(self) -> None:
        quoted = re.search(r"(\d+\.\d+)%", _floorComment())
        assert quoted is not None, "the comment above COVERAGE_FLOOR quotes no measured value"
        gap = float(quoted.group(1)) - _floor()
        assert 1.0 <= gap < 2.0, (
            f"the comment quotes {quoted.group(1)}% and the floor is {_floor()}: the floor is "
            f"either not one point below the value or the comment is stale"
        )

    def test_the_readme_and_the_getting_started_page_state_the_same_floor(self) -> None:
        for path in (_REPO_ROOT / "README.md", _REPO_ROOT / "docs" / "getting-started.md"):
            text = path.read_text(encoding="utf-8")
            commands = re.findall(r"--cov-fail-under=(\d+)", text)
            assert commands == [str(_floor())], f"{path.name}: {commands}"
            prose = re.findall(r"\b(\d+)% (?:of statements )?floor|floor is (\d+)%", text)
            numbers = {int(first or second) for first, second in prose}
            assert numbers == {_floor()}, f"{path.name}: {numbers}"


class TestTheTestsJob:
    def test_it_reports_its_slowest_tests(self) -> None:
        assert any("--durations=15" in line for line in _jobBlock("tests"))

    def test_it_keeps_a_timeout_that_the_suite_does_not_come_near(self) -> None:
        limits = [
            int(match.group(1))
            for line in _jobBlock("tests")
            if (match := re.match(r"^    timeout-minutes:\s*(\d+)\s*$", line))
        ]
        assert limits == [30]

    def test_the_gated_run_does_not_trace_branches(self) -> None:
        # The floor is a percentage of statements. Branch tracing in the gated run would change
        # the number the floor is compared with, and the floor is not recomputed for it.
        assert "--cov-branch" not in "\n".join(_jobBlock("tests"))


class TestTheBranchCoverageJob:
    def test_the_job_exists(self) -> None:
        assert _BRANCH_JOB in _jobHeaders()

    def test_it_runs_on_the_schedule_and_on_a_manual_dispatch_only(self) -> None:
        conditions = [line for line in _jobBlock(_BRANCH_JOB) if re.match(r"^    if:", line)]
        assert len(conditions) == 1
        assert "schedule" in conditions[0] and "workflow_dispatch" in conditions[0]
        assert "push" not in conditions[0] and "pull_request" not in conditions[0]

    def test_it_traces_branches_and_gates_on_nothing(self) -> None:
        block = "\n".join(_jobBlock(_BRANCH_JOB))
        assert "--cov-branch" in block
        assert "--cov-fail-under" not in block
        assert "fail_under" not in block

    def test_it_reports_the_three_packages_to_the_summary_of_the_run(self) -> None:
        block = "\n".join(_jobBlock(_BRANCH_JOB))
        assert "GITHUB_STEP_SUMMARY" in block
        for package in ("latent", "models", "training"):
            assert f"*/global_vae/{package}/*" in block
