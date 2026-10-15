"""The Hugging Face Space bundle: what gets uploaded and the safe defaults baked into it."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("deploy_hf", ROOT / "scripts" / "deploy_hf.py")
assert spec and spec.loader
deploy_hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy_hf)

needs_recording = pytest.mark.skipif(
    not (ROOT / "data" / "demo_answers.json").exists(), reason="demo answers not recorded yet"
)


def files(dest: Path) -> set[str]:
    return {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}


@needs_recording
def test_bundle_contains_only_what_the_image_needs(tmp_path: Path) -> None:
    deploy_hf.build_bundle(tmp_path)
    names = files(tmp_path)
    assert {"Dockerfile", "README.md", "pyproject.toml", "data/demo_answers.json"} <= names
    assert "src/docrag/api.py" in names and "src/docrag/static/index.html" in names
    assert any(n.startswith("data/samples/") for n in names)
    for n in names:  # nothing private, generated or irrelevant to the running app
        assert not n.startswith(("data/chroma", "data/raw", "eval/", "tests/", ".git", "docs/"))
        assert ".env" not in Path(n).name and not n.endswith((".pyc", ".gif"))


@needs_recording
def test_space_card_selects_docker_on_the_app_port(tmp_path: Path) -> None:
    deploy_hf.build_bundle(tmp_path)
    card = (tmp_path / "README.md").read_text()
    assert card.startswith("---\n") and "sdk: docker" in card and "app_port: 8000" in card
    assert len(next(ln for ln in card.splitlines() if ln.startswith("short_description:"))) < 90
    assert "recorded" in card.lower()


@needs_recording
def test_image_is_the_replay_demo_with_no_key_and_a_non_root_user(tmp_path: Path) -> None:
    deploy_hf.build_bundle(tmp_path)
    dockerfile = (tmp_path / "Dockerfile").read_text()
    assert dockerfile == (ROOT / "Dockerfile.demo").read_text()
    for needle in ("DEMO_REPLAY=true", "DEMO_MODE=true", "USER user", "docrag.ingest"):
        assert needle in dockerfile
    assert "API_KEY" not in dockerfile and "LLM_PROVIDER" not in dockerfile


def test_building_without_a_recording_fails_with_a_clear_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_root = tmp_path / "repo"
    (fake_root / "data" / "samples").mkdir(parents=True)
    monkeypatch.setattr(deploy_hf, "ROOT", fake_root)
    with pytest.raises(SystemExit, match="record_demo"):
        deploy_hf.build_bundle(tmp_path / "out")


@needs_recording
def test_no_secret_is_ever_written_into_the_bundle(tmp_path: Path) -> None:
    deploy_hf.build_bundle(tmp_path)
    text = [
        p
        for p in tmp_path.rglob("*")
        if p.is_file() and p.suffix in {".py", ".md", ".toml", ".html", ""}
    ]
    assert not any("sk-" in p.read_text(errors="ignore") for p in text)
    assert not any(p.name == ".env" for p in tmp_path.rglob("*"))
