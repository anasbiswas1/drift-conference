"""Single commit step for this repo. Run from a notebook's final cell only:

    subprocess.run(["python", "tools/commit_cell.py", "message"])

Restores git identity and credentials from the Drive folder, strips the outputs of every notebook that
git reports as changed, refuses to stage data, credential, parquet or oversized files, refuses any staged
file containing a secret token pattern, commits if anything is staged, and always pushes so unpushed
commits never remain local.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(os.environ.get("DRIFT_REPO_ROOT", "/content/drive/MyDrive/drift-conference"))
DRIVE_PROJECT_ROOT = REPO_ROOT                      # the Drive folder is the repo root in this project
GIT_NAME = "Md Anas Biswas"
GIT_EMAIL = "anasbiswas@gmail.com"
FORBIDDEN_NAMES = {".git-credentials", ".gitconfig", "kaggle.json", ".env"}
FORBIDDEN_PREFIXES = ("data/", ".secrets/")
FORBIDDEN_SUFFIXES = (".parquet", ".npy", ".pkl")
LOCAL_EXCLUDES = (".git-credentials", ".gitconfig", "/*.ipynb")   # root-level scratch notebooks are never staged
MAX_FILE_MB = 50
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),            # Hugging Face
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),           # GitHub classic PAT
    re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),   # GitHub fine-grained PAT
    re.compile(r"sk-[A-Za-z0-9]{32,}"),             # OpenAI-style keys
    re.compile(r"AKIA[0-9A-Z]{16}"),               # AWS access key id
    re.compile(r"AIza[0-9A-Za-z_\-]{35}"),         # Google API key
]
TEXT_EXT = {".py", ".ipynb", ".md", ".txt", ".json", ".csv", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".sh"}


def run(args, check=True):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"{' '.join(args)}\n{r.stdout}{r.stderr}")
    return r.stdout


def restore_credentials():
    """Prefer this repo's own dotfiles; fall back to any project folder on this Drive account."""
    for dotfile in (".gitconfig", ".git-credentials"):
        source = DRIVE_PROJECT_ROOT / dotfile
        if not source.exists() and dotfile == ".git-credentials":
            found = glob.glob("/content/drive/MyDrive/*/.git-credentials")
            source = Path(found[0]) if found else source
        if source.exists():
            shutil.copy(source, Path.home() / dotfile)
    credentials = Path.home() / ".git-credentials"
    if credentials.exists():
        os.chmod(credentials, 0o600)


def ensure_local_excludes():
    exclude = REPO_ROOT / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    present = exclude.read_text().splitlines() if exclude.exists() else []
    missing = [e for e in LOCAL_EXCLUDES if e not in present]
    if missing:
        with exclude.open("a") as f:
            f.write("\n".join(missing) + "\n")


def changed_notebooks():
    """Notebooks that are modified, added or untracked. Committed notebooks are already stripped."""
    paths = []
    for line in run(["git", "status", "--porcelain", "--untracked-files=all", "--", "notebooks"]).splitlines():
        p = line[3:].strip().strip('"')
        if " -> " in p:
            p = p.split(" -> ")[-1]
        if p.endswith(".ipynb") and (REPO_ROOT / p).is_file():
            paths.append(REPO_ROOT / p)
    return sorted(set(paths))


def strip_outputs():
    import nbformat
    notebooks = changed_notebooks()
    for notebook in notebooks:
        nb = nbformat.read(notebook, as_version=4)
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        nb.metadata.pop("widgets", None)
        nbformat.write(nb, notebook)
    print(f"outputs: stripped {len(notebooks)} changed notebook(s)")


def secret_scan(paths):
    hits = []
    for p in paths:
        path = REPO_ROOT / p
        if path.suffix.lower() not in TEXT_EXT or not path.is_file():
            continue
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat in SECRET_PATTERNS:
            if pat.search(txt):
                hits.append(f"{p} ({pat.pattern[:14]}...)")
                break
    return hits


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        raise SystemExit('usage: python tools/commit_cell.py "commit message"')
    message = sys.argv[1].strip()

    restore_credentials()
    os.chdir(REPO_ROOT)
    run(["git", "config", "user.name", GIT_NAME])
    run(["git", "config", "user.email", GIT_EMAIL])
    run(["git", "config", "credential.helper", "store"])
    run(["git", "config", "core.logAllRefUpdates", "false"])   # Drive refuses writes to git's reflog file
    ensure_local_excludes()

    strip_outputs()

    run(["git", "add", "-A"])
    staged = [l for l in run(["git", "diff", "--cached", "--name-only"]).splitlines() if l.strip()]
    bad = [p for p in staged if Path(p).name in FORBIDDEN_NAMES or p.startswith(FORBIDDEN_PREFIXES)
           or p.endswith(FORBIDDEN_SUFFIXES)]
    big = [p for p in staged if Path(p).exists() and Path(p).stat().st_size > MAX_FILE_MB * 1024 * 1024]
    secrets = secret_scan(staged)
    if bad or big or secrets:
        run(["git", "reset", "-q"], check=False)
        raise SystemExit(f"refused: forbidden={bad} oversized={big} secrets_in={secrets}")

    if staged:
        mb = sum(Path(p).stat().st_size for p in staged if Path(p).exists()) / 1e6
        print(f"staged {len(staged)} files, {mb:.1f} MB")
        commit = subprocess.run(["git", "commit", "-m", message], capture_output=True, text=True)
        print(commit.stdout or commit.stderr)
    else:
        print("nothing new to commit; pushing any unpushed commits")

    push = subprocess.run(["git", "push", "origin", "HEAD:main"], capture_output=True, text=True)
    print(push.stdout or push.stderr)
    print(run(["git", "log", "--oneline", "-1"]).strip())
    print(run(["git", "status", "--short"]).strip() or "working tree clean")


if __name__ == "__main__":
    main()
