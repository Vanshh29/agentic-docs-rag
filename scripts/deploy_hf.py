"""Publish the read-only demo to a Hugging Face Docker Space.

Usage:
    python scripts/deploy_hf.py --dry-run                 # only build the bundle and list it
    python scripts/deploy_hf.py --space USER/NAME                     # create/update, stays private
    python scripts/deploy_hf.py --space USER/NAME --make-public       # when you are happy with it

Needs `huggingface-cli login` with a token that can write (role "write", or a fine-grained token
with repo write access). The Space replays recorded answers (data/demo_answers.json, made by
scripts/record_demo.py), so it needs no LLM key and makes no model calls.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPACE_CARD = """---
title: Docs Assistant
sdk: docker
app_port: 8000
pinned: false
short_description: Ask questions about technical manuals, with cited answers
---

# Docs Assistant

Question answering over technical manuals (PDF and DOCX, German and English). Every answer cites
the page it came from, and the assistant says so when the documents do not cover a question.

This is a read-only recorded demo: the example questions were run once against three invented
sample manuals, and their real answers are replayed here, so no model is called and no API key
is involved. To ask your own questions, run the project locally.
Source code and evaluation: https://github.com/Vanshh29/agentic-docs-rag
"""

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", ".DS_Store")


def build_bundle(dest: Path) -> None:
    """Assemble the minimal build context for the Space in `dest`."""
    answers = ROOT / "data" / "demo_answers.json"
    if not answers.exists():
        raise SystemExit("data/demo_answers.json is missing: run scripts/record_demo.py first")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "Dockerfile.demo", dest / "Dockerfile")
    (dest / "README.md").write_text(SPACE_CARD)
    shutil.copy(ROOT / "pyproject.toml", dest / "pyproject.toml")
    shutil.copytree(ROOT / "src", dest / "src", ignore=_IGNORE, dirs_exist_ok=True)
    shutil.copytree(
        ROOT / "data" / "samples", dest / "data" / "samples", ignore=_IGNORE, dirs_exist_ok=True
    )
    shutil.copy(answers, dest / "data" / "demo_answers.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--space", help="USER/NAME of the Space (default: <your user>/docs-assistant-demo)"
    )
    p.add_argument("--make-public", action="store_true", help="switch the Space to public")
    p.add_argument("--dry-run", action="store_true", help="build the bundle and stop")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        build_bundle(bundle)
        if args.dry_run:
            for f in sorted(x.relative_to(bundle) for x in bundle.rglob("*") if x.is_file()):
                print(f)
            return

        from huggingface_hub import HfApi

        api = HfApi()
        repo_id = args.space or f"{api.whoami()['name']}/docs-assistant-demo"
        # Created private: nothing is public until --make-public.
        api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(bundle),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy read-only demo",
        )
        if args.make_public:
            api.update_repo_settings(repo_id, repo_type="space", private=False)
        info = api.space_info(repo_id)
        print(f"Space: https://huggingface.co/spaces/{repo_id}  (private: {info.private})")
        print("The Space builds automatically; open the URL above to follow the build.")


if __name__ == "__main__":
    main()
