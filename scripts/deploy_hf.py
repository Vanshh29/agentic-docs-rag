"""Publish the read-only demo to a Hugging Face Docker Space.

Usage:
    python scripts/deploy_hf.py --dry-run                 # only build the bundle and list it
    python scripts/deploy_hf.py --space USER/NAME --model MODEL_ID    # create/update, stays private
    python scripts/deploy_hf.py --space USER/NAME --make-public       # when you are happy with it

Needs `huggingface-cli login` with a token that can write (role "write", or a fine-grained token
with repo write access). The LLM key is never uploaded: add OPENROUTER_API_KEY yourself under
Space -> Settings -> Variables and secrets.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Baked into the image so a public Space can never start with uploads or deletes enabled.
DEMO_ENV = """
# --- public demo (added by scripts/deploy_hf.py) ---
RUN useradd -m -u 1000 user && mkdir -p data/chroma data/raw && chown -R user:user /app
USER user
ENV DEMO_MODE=true PRELOAD=true ASK_RATE_PER_MINUTE=5 ASK_DAILY_LIMIT=200 LLM_PROVIDER=openrouter
"""

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

This is a read-only demo over three invented sample manuals, with a per-visitor rate limit.
Source code and evaluation: https://github.com/Vanshh29/agentic-docs-rag
"""

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", ".DS_Store")


def build_bundle(dest: Path) -> None:
    """Assemble the minimal build context for the Space in `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    dockerfile = (ROOT / "Dockerfile").read_text().rstrip("\n") + "\n" + DEMO_ENV
    (dest / "Dockerfile").write_text(dockerfile)
    (dest / "README.md").write_text(SPACE_CARD)
    shutil.copy(ROOT / "pyproject.toml", dest / "pyproject.toml")
    shutil.copytree(ROOT / "src", dest / "src", ignore=_IGNORE, dirs_exist_ok=True)
    shutil.copytree(
        ROOT / "data" / "samples", dest / "data" / "samples", ignore=_IGNORE, dirs_exist_ok=True
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--space", help="USER/NAME of the Space (default: <your user>/docs-assistant-demo)"
    )
    p.add_argument("--model", help="value for the LLM_MODEL variable (a model id for the provider)")
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
        if args.model:
            api.add_space_variable(repo_id, "LLM_MODEL", args.model)
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
        print("Next: add the secret OPENROUTER_API_KEY under Settings -> Variables and secrets.")


if __name__ == "__main__":
    main()
