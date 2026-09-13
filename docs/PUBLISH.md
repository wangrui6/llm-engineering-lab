# Publish this starter to GitHub

This archive is prepared for `wangrui6/llm-engineering-lab`. Preparing this archive does **not** mean that repository already exists. GitHub and Colab links in the README become usable after the public push.

## First publication

Unzip into a fresh folder outside any existing Git repository. Review the files and the MIT license. The archive contains teaching code and original instructional material, not the unpublished blog manuscript or private Notion links.

You need Python 3.10+, Git, and [GitHub CLI](https://cli.github.com/). Install the CLI using its official instructions. On a Mac with Homebrew, the installation command is `brew install gh`.

From the repository root:

```bash
# Browser-based login. Never paste a token into an AI chat.
gh auth login --hostname github.com --web --git-protocol https --scopes workflow

# Offline preview: lists the exact public file allowlist.
python3 scripts/publish.py

# This command creates a PUBLIC repository and pushes the code.
python3 scripts/publish.py --publish
```

The extra `workflow` scope allows the OAuth credential to push the GitHub Actions workflow file. Existing authenticated CLI users may need `gh auth refresh --hostname github.com --scopes workflow` instead of a second login.

The helper checks the authenticated owner, refuses an existing target repository, runs tests, configures Git's GitHub credential helper through `gh auth setup-git`, and creates one initial commit. It stages only files listed in `scripts/public-files.json`. It uses the account's GitHub no-reply address for this commit without changing your global Git identity. It then creates and pushes a public repository, verifies the remote commit and visibility, and attempts to enable topics, a template button, Issues, and Discussions.

No deletion or force push is performed. No paid service is provisioned. MIT is the included license choice; review or replace it before publication. Files you add after extracting are not automatically staged. Add an intended public file to the allowlist only after reviewing its contents.

## Verify after the push

Open the repository without signing in and check the README and ZIP download. Open the Actions tab and confirm the jobs passed. Open the Colab button in a fresh session, run the setup, and solve one exercise. Hosted CI and the GitHub-to-Colab path cannot be validated before the repository is published.

The notebook also works by direct upload to Colab. It is self-contained and does not need to clone this repository. The local CLI path remains the dependency-free fallback.

## What the helper runs underneath

After a reviewed local commit, the central operation is:

```bash
gh repo create wangrui6/llm-engineering-lab   --public --source=. --remote=origin --push
```

GitHub CLI officially supports creating a repository from a local source and pushing its commits. [Create documentation](https://cli.github.com/manual/gh_repo_create). Template, topics, and Discussions are separate settings. [Edit documentation](https://cli.github.com/manual/gh_repo_edit). [Authentication](https://cli.github.com/manual/gh_auth_login) and [Git credential setup](https://cli.github.com/manual/gh_auth_setup-git) explain the login behavior.

## Recover from an interrupted first push

Do not delete the remote or force-push. The helper deliberately refuses a folder with existing Git history on a second run. Inspect:

```bash
git status
git remote -v
git log -1 --oneline
gh repo view wangrui6/llm-engineering-lab
```

A network failure can leave a local commit, a new empty remote, or a completed push whose verification failed. Match the remote URL and commit before retrying a normal `git push -u origin main`. If the remote was never created, use the central `gh repo create` command above from the reviewed local repository. If the target name is already occupied by a different project, stop and choose another name; update README, citation, and Colab links deliberately.

For future edits, use ordinary branches and pull requests. This helper is only for the first publication.
