"""The demo recorder's guard against product names, and the committed recording itself."""

import importlib.util
import json
from pathlib import Path

import pytest

from docrag.replay import ReplayBook, normalize

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("record_demo", ROOT / "scripts" / "record_demo.py")
assert spec and spec.loader
record_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(record_demo)
ANSWERS = ROOT / "data" / "demo_answers.json"


def test_guard_reads_blocked_terms_from_the_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocklist = tmp_path / "blocklist"
    blocklist.write_text("# comment\nAlpha\n\nbeta\n")
    monkeypatch.setattr(record_demo, "BLOCKLIST", blocklist)
    assert record_demo.find_blocked_terms("An ALPHA and a Beta thing") == ["Alpha", "beta"]
    assert record_demo.find_blocked_terms("Descale every 3 months [1].") == []


def test_guard_is_a_no_op_without_the_local_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(record_demo, "BLOCKLIST", tmp_path / "missing")
    assert record_demo.find_blocked_terms("anything at all") == []


def test_question_list_has_unique_questions_in_both_outcomes() -> None:
    keys = [normalize(q) for q, _ in record_demo.QUESTIONS]
    assert len(keys) == len(set(keys))
    assert {should for _, should in record_demo.QUESTIONS} == {True, False}


@pytest.mark.skipif(not ANSWERS.exists(), reason="demo answers not recorded yet")
def test_committed_recording_is_loadable_complete_and_clean() -> None:
    book = ReplayBook.load(ANSWERS)
    for question, should_answer in record_demo.QUESTIONS:
        recorded = book.get(question)
        assert recorded is not None, question
        assert recorded.declined is (not should_answer), question
        if should_answer:
            assert recorded.citations, question
    assert record_demo.find_blocked_terms(ANSWERS.read_text()) == []
    assert json.loads(ANSWERS.read_text())["recorded_at"]
