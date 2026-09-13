"""Publish this reviewed starter as a NEW PUBLIC repository on GitHub.

Default: show an offline preview. With --publish: verify the account, run
checks, create one initial commit, create a public repository, push it, then
verify the remote commit. Existing repositories and Git history are refused.
No force pushes, deletes, paid services, or token collection are used.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OWNER = "wangrui6"
NAME = "llm-engineering-lab"
TARGET = OWNER + "/" + NAME
DESCRIPTION = "Learn LLM engineering through small, tested exercises. Start with retrieval: BM25, sparse scoring, MaxSim, RRF, and evaluation."


def public_files():
    files = json.loads((ROOT / "scripts/public-files.json").read_text(encoding="utf-8"))
    if not isinstance(files, list) or not files or any(not isinstance(f, str) for f in files):
        raise ValueError("Invalid public file allowlist")
    if len(files) != len(set(files)):
        raise ValueError("Duplicate allowlist entry")
    for name in files:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
            raise ValueError("Unsafe allowlist path: " + name)
        p = ROOT / name
        if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(ROOT):
            raise ValueError("Missing or unsafe public file: " + name)
    return files


def run(args, *, capture=False, check=True):
    env = dict(os.environ, GH_HOST="github.com")
    result = subprocess.run(args, cwd=ROOT, env=env, text=True,
                            capture_output=capture, check=False)
    if check and result.returncode:
        detail = (result.stderr or result.stdout or "").strip() if capture else "See command output above."
        raise RuntimeError("Command failed: " + " ".join(args) + "\n" + detail)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="Create and push the PUBLIC repository; otherwise only preview")
    args = parser.parse_args()
    files = public_files()
    print("Target: " + TARGET, flush=True)
    print("Visibility: PUBLIC. License included: MIT. Review LICENSE and all listed files before publishing.", flush=True)
    print("Public files: " + str(len(files)), flush=True)
    for name in files:
        print("  " + name, flush=True)
    if not args.publish:
        print("\nPreview only: no login, Git changes, network access, or publication performed.")
        print("To publish after review: python3 scripts/publish.py --publish")
        return 0
    for tool in ("git", "gh"):
        if shutil.which(tool) is None:
            raise RuntimeError(tool + " is not installed. See docs/PUBLISH.md. Nothing has been uploaded.")
    # Do not accidentally publish an existing repo's history or a parent project.
    probe = run(["git", "rev-parse", "--show-toplevel"], capture=True, check=False)
    if probe.returncode == 0 or (ROOT / ".git").exists():
        raise RuntimeError("Use a fresh extracted folder outside another Git repository. Existing Git history is not handled by this first-publication helper. See docs/PUBLISH.md for recovery.")
    identity = json.loads(run(["gh", "api", "--hostname", "github.com", "user",
                               "--jq", "{login: .login, id: .id}"], capture=True).stdout)
    if identity["login"] != OWNER:
        raise RuntimeError("Authenticated account is " + identity["login"] + "; expected " + OWNER + ". Switch accounts before publishing.")
    existing = run(["gh", "api", "--hostname", "github.com", "repos/" + TARGET], capture=True, check=False)
    if existing.returncode == 0:
        raise RuntimeError("Repository already exists. Refusing to overwrite or append to it.")
    if "HTTP 404" not in existing.stderr:
        raise RuntimeError("Could not safely establish that the target name is unused. Resolve the API/network error first.\n" + existing.stderr)
    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    run([sys.executable, "scripts/verify_notebook.py"])
    # gh configures GitHub's credential helper. No token is passed in arguments.
    run(["gh", "auth", "setup-git", "--hostname", "github.com"])
    run(["git", "init", "-b", "main"])
    run(["git", "add", "--"] + files)  # Explicit allowlist; never `git add .`.
    run(["git", "diff", "--cached", "--stat"])
    noreply = str(identity["id"]) + "+" + OWNER + "@users.noreply.github.com"
    run(["git", "-c", "user.name=Rui Wang", "-c", "user.email=" + noreply,
         "commit", "-m", "Add retrieval mechanics lab with exercises, tests, and notebook"])
    run(["gh", "repo", "create", TARGET, "--public", "--source=.",
         "--remote=origin", "--push", "--description", DESCRIPTION])
    local_sha = run(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    remote_sha = run(["gh", "api", "--hostname", "github.com", "repos/" + TARGET + "/commits/main",
                      "--jq", ".sha"], capture=True).stdout.strip()
    metadata = json.loads(run(["gh", "api", "--hostname", "github.com", "repos/" + TARGET,
                               "--jq", "{url: .html_url, private: .private}"], capture=True).stdout)
    if remote_sha != local_sha or metadata["private"]:
        raise RuntimeError("Repository creation returned, but commit/visibility verification failed. Inspect GitHub before making further changes.")
    topics = ("llm-engineering", "information-retrieval", "bm25", "colbert", "rag", "python", "education")
    edit = ["gh", "repo", "edit", TARGET, "--default-branch", "main", "--template",
            "--enable-issues", "--enable-discussions"]
    for topic in topics:
        edit.extend(["--add-topic", topic])
    result = run(edit, check=False)
    print("\nPUBLIC PUSH VERIFIED: " + metadata["url"])
    print("Commit: " + local_sha)
    if result.returncode:
        print("Optional topics/template/discussion settings failed; the public code push was verified. Apply those settings in GitHub's UI.")
    print("Check the Actions tab: local checks do not prove that hosted CI has passed.")
    print("Then verify the Colab button in a fresh browser session and pin the repository on your profile.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError, KeyError) as error:
        print("\nSTOPPED: " + str(error), file=sys.stderr)
        print("No existing remote repository was intentionally overwritten. A failed network operation may leave a local commit or a newly created repository; inspect before retrying.", file=sys.stderr)
        sys.exit(1)
