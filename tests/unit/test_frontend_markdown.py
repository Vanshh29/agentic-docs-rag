"""The answer renderer's parser (pure JS in index.html) is exercised through Node."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HTML = Path(__file__).resolve().parents[2] / "src/docrag/static/index.html"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def parse(text: str) -> list[dict]:
    src = HTML.read_text()
    m = re.search(r"// <parse-answer>(.*?)// </parse-answer>", src, re.S)
    assert m, "parse-answer markers missing from index.html"
    script = m.group(1) + f"\nconsole.log(JSON.stringify(parseAnswer({json.dumps(text)})));"
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)  # type: ignore[no-any-return]


def kinds(inlines: list[dict]) -> list[str]:
    return [i["t"] for i in inlines]


def test_bold_and_citation_markers_become_tokens() -> None:
    (p,) = parse("**Heat-up time:** about 25 seconds [1].")
    assert p["type"] == "p"
    assert kinds(p["inlines"]) == ["bold", "text", "cite", "text"]
    assert p["inlines"][0]["v"] == "Heat-up time:" and p["inlines"][2]["v"] == 1


def test_bullets_group_into_one_list_and_paragraphs_split_on_blank_lines() -> None:
    blocks = parse("Other things:\n- **Dose:** 8 g [1]\n- Rinse daily [3]\n\nCall support [2].")
    assert [b["type"] for b in blocks] == ["p", "ul", "p"]
    assert len(blocks[1]["items"]) == 2
    assert kinds(blocks[1]["items"][0]) == ["bold", "text", "cite"]


def test_numbered_lists_and_star_bullets() -> None:
    blocks = parse("1. Switch off\n2) Unplug\n\n* one\n* two")
    assert [b["type"] for b in blocks] == ["ol", "ul"]
    assert len(blocks[0]["items"]) == 2 and len(blocks[1]["items"]) == 2


def test_headings_are_kept_as_emphasis_without_the_hashes() -> None:
    (h, p) = parse("## Descaling\nEvery 3 months.")
    assert h["type"] == "h" and h["inlines"][0]["v"] == "Descaling"
    assert p["type"] == "p"


def test_html_is_never_interpreted_it_stays_plain_text() -> None:
    (p,) = parse("<img src=x onerror=alert(1)> **ok**")
    assert p["inlines"][0] == {"t": "text", "v": "<img src=x onerror=alert(1)> "}
    assert p["inlines"][1] == {"t": "bold", "v": "ok"}


def test_plain_text_and_soft_line_breaks_survive() -> None:
    (p,) = parse("line one\nline two")
    assert "".join(i["v"] for i in p["inlines"] if i["t"] == "text") == "line one\nline two"


def test_unclosed_bold_and_stray_asterisks_stay_literal() -> None:
    (p,) = parse("5 * 3 and **unclosed")
    assert kinds(p["inlines"]) == ["text"]
    assert p["inlines"][0]["v"] == "5 * 3 and **unclosed"
