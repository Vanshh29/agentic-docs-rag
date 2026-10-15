"""The Hugging Face Space bundle: what gets uploaded and the safe defaults baked into it."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("deploy_hf", ROOT / "scripts" / "deploy_hf.py")
assert spec and spec.loader
deploy_hf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy_hf)


def files(dest: Path) -> set[str]:
    return {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}


def test_bundle_contains_only_what_the_image_needs(tmp_path: Path) -> None:
    deploy_hf.build_bundle(tmp_path)
    names = files(tmp_path)
    assert {"Dockerfile", "README.md", "pyproject.toml"} <= names
    assert "src/docrag/api.py" in names and "src/docrag/static/index.html" in names
    assert any(n.startswith("data/samples/") for n in names)
    for n in names:  # nothing private, generated or irrelevant to the running app
        assert not n.startswith(("data/chroma", "data/raw", "eval/", "tests/", ".git", "docs/"))
        assert ".env" not in Path(n).name and not n.endswith((".pyc", ".gif"))


def test_space_card_selects_docker_on_the_app_port(tmp_path: Path) -> None:
    deploy_hf.build_bundle(tmp_path)
    card = (tmp_path / "README.md").read_text()
    assert card.startswith("---\n") and "sdk: docker" in card and "app_port: 8000" in card
    assert len(next(ln for ln in card.splitlines() if ln.startswith("short_description:"))) < 90


def test_dockerfile_bakes_in_safe_public_defaults_and_a_non_root_user(tmp_path: Path) -> None:
    deploy_hf.build_bundle(tmp_path)
    dockerfile = (tmp_path / "Dockerfile").read_text()
    for needle in ("DEMO_MODE=true", "PRELOAD=true", "ASK_RATE_PER_MINUTE=", "ASK_DAILY_LIMIT="):
        assert needle in dockerfile
    assert "USER user" in dockerfile and "chown -R user:user /app" in dockerfile
    original = (ROOT / "Dockerfile").read_text()
    assert dockerfile.startswith(original.rstrip("\n"))  # the repo's own Dockerfile is the base
    assert "DEMO_MODE" not in original  # and it is not modified in place


def test_no_secret_is_ever_written_into_the_bundle(tmp_path: Path) -> None:
    deploy_hf.build_bundle(tmp_path)
    blob = "\n".join(
        p.read_text(errors="ignore")
        for p in tmp_path.rglob("*")
        if p.is_file() and p.suffix in {".py", ".md", ".toml", "", ".html"}
    )
    assert "OPENROUTER_API_KEY=" not in blob.replace("OPENROUTER_API_KEY=\n", "")
