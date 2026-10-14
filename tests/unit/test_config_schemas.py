import pytest
from pydantic import ValidationError

from docrag.schemas import AskRequest
from tests.fakes import make_settings


def test_question_is_stripped() -> None:
    assert AskRequest(question="  hi there \n").question == "hi there"


@pytest.mark.parametrize("q", ["", "   ", "\n\t"])
def test_blank_question_rejected(q: str) -> None:
    with pytest.raises(ValidationError):
        AskRequest(question=q)


def test_question_max_length() -> None:
    assert len(AskRequest(question="x" * 2000).question) == 2000
    with pytest.raises(ValidationError):
        AskRequest(question="x" * 2001)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    assert make_settings(chunk_size=200, chunk_overlap=199).chunk_overlap == 199
    for size, overlap in [(200, 200), (200, 300)]:
        with pytest.raises(ValidationError, match="chunk_overlap .* must be smaller"):
            make_settings(chunk_size=size, chunk_overlap=overlap)


def test_env_file_is_ignored_by_helper(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / ".env").write_text("TOP_K=99\n")
    monkeypatch.chdir(tmp_path)
    assert make_settings().top_k == 6
