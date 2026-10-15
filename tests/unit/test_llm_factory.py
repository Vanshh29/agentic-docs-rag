"""A missing provider key must not take the whole app down at start-up."""

import logging

import pytest

from docrag.llm import make_llm
from tests.fakes import make_settings


@pytest.mark.parametrize(
    ("provider", "key_field"),
    [
        ("openrouter", "openrouter_api_key"),
        ("openai", "openai_api_key"),
        ("anthropic", "anthropic_api_key"),
    ],
)
def test_missing_api_key_warns_instead_of_crashing(
    provider: str, key_field: str, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    settings = make_settings(llm_provider=provider, llm_model="some/model", **{key_field: ""})
    with caplog.at_level(logging.WARNING, logger="docrag.llm"):
        llm = make_llm(settings)
    assert llm is not None
    assert any(key_field.upper() in r.message for r in caplog.records)
