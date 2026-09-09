"""Commit helper for the drift-conference repo.

Usage (from the scratch console notebook, never from numbered notebooks):
    subprocess.run(["python", "tools/commit_cell.py", "message"])
    subprocess.run(["python", "tools/commit_cell.py", "message", "path1", "path2"])

With no paths: stages every tracked change plus new files under notebooks/,
results/, tools/. With paths: stages only those. Always strips notebook
outputs first, refuses data/credential files, and prints proof.
"""
import glob
import os
import shutil
import subprocess
import sys

REPO = "/content/drive/MyDrive/drift-conference"
DEFAULT_DIRS = ["notebooks", "results", "tools"]
BLOCKED_SUFFIX = (".parquet", ".json")
BLOCKED_PREFIX = ("data/",)
ALLOWED_JSON = ("results/nfv2/corpora_manifest.json", "manifests/dataset_manifest.json")


def run(cmd, check=True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True).stdout


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: commit_cell.py <message> [paths...]")
    msg, paths = sys.argv[1], sys.argv[2:]
    os.chdir(REPO)

    cred = glob.glob("/content/drive/MyDrive/*/.git-credentials")
    if cred:
        shutil.copy(cred[0], "/root/.git-credentials")
    run(["git", "config", "--global", "credential.helper", "store"])
    run(["git", "config", "--global", "user.name", "Md Anas Biswas"])
    run(["git", "config", "--global", "user.email", "anasbiswas@gmail.com"])

    for nb in sorted(glob.glob("notebooks/*.ipynb")):
        subprocess.run(["jupyter", "nbconvert", "--ClearOutputPreprocessor.enabled=True",
                        "--inplace", nb], capture_output=True)

    if paths:
        for p in paths:
            assert os.path.exists(p), f"missing: {p}"
            run(["git", "add", "-f", p])
    else:
        run(["git", "add", "-u"])
        for d in DEFAULT_DIRS:
            if os.path.isdir(d):
                run(["git", "add", d])

    staged = run(["git", "diff", "--cached", "--name-only"]).split()
    for s in staged:
        blocked = (s.endswith(BLOCKED_SUFFIX) and s not in ALLOWED_JSON) or s.startswith(BLOCKED_PREFIX)
        assert not blocked, f"blocked file staged: {s}"
    if paths:
        missing = [p for p in paths if p not in staged]
        assert not missing, f"not staged: {missing}"
    if not staged:
        print("nothing to commit")
        return
    mb = sum(os.path.getsize(s) for s in staged if os.path.isfile(s)) / 1e6
    assert mb < 200, f"staged size {mb:.1f} MB exceeds limit"
    print(f"staged {len(staged)} files, {mb:.1f} MB")

    run(["git", "commit", "-m", msg])
    run(["git", "push"])
    print(run(["git", "log", "--oneline", "-1"]).strip())
    print(run(["git", "status", "--short"]).strip() or "working tree clean")


if __name__ == "__main__":
    main()
