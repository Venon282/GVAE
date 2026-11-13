"""Every item of the roadmap says whether it is done (docs/roadmap.md, section 1)."""

import re
from pathlib import Path

import pytest

_ROADMAP = Path(__file__).resolve().parents[2] / "docs" / "roadmap.md"
_HEADING = re.compile(r"^### (P\d-\d+)\.")
_STATUS = re.compile(r"^\*\*status\*\*: (todo|done)$")


def _items() -> dict[str, list[str]]:
    """Item id -> the lines between its heading and the next heading."""
    items: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in _ROADMAP.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            match = _HEADING.match(line)
            current = items.setdefault(match.group(1), []) if match else None
        elif current is not None:
            current.append(line)
    return items


class TestTheStatusOfTheItems:
    def test_the_roadmap_has_its_items(self) -> None:
        items = _items()
        assert {"P5-3", "P6-1", "P7-1", "P4-0"} <= set(items)
        assert len(items) >= 27

    @pytest.mark.parametrize("item", sorted(_items()))
    def test_an_item_has_one_status_line_under_its_heading(self, item: str) -> None:
        lines = _items()[item]
        statuses = [line for line in lines if line.startswith("**status**")]
        assert len(statuses) == 1, f"{item}: expected one `**status**: todo|done` line"
        assert _STATUS.match(statuses[0]), f"{item}: {statuses[0]!r} is not todo or done"
        first = next(line for line in lines if line.strip())
        assert first == statuses[0], f"{item}: the status line comes right after the heading"

    def test_the_items_delivered_so_far_are_marked_done(self) -> None:
        items = _items()
        for item in ("P5-1", "P5-2", "P5-3", "P5-4", "P5-5", "P5-6", "P5-7", "P6-1", "P6-2"):
            assert "**status**: done" in items[item], item
