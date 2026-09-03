#!/usr/bin/env bash
# Creates a git worktree for digitizing one chapter, wired up so the scripts run.
#
# Usage:
#     utils/worktree.sh turbocharger-system          # create ../mr2-turbocharger-system
#     utils/worktree.sh --remove turbocharger-system # remove it again
#
# Three things the repository needs are gitignored, so a plain `git worktree add`
# produces a tree where every script fails:
#   .venv                        - all documented commands are .venv/bin/python ...
#   .staging                     - 0.75 MB and ~3 s per page to re-render; share it
#   .claude/settings.local.json  - holds MR2_DOCS_MANUAL_PATH; without it the tracked
#                                  placeholder wins and the scripts exit "manual not found"
#
# The .venv symlink is kept at the worktree root on purpose: the permission rules in
# .claude/settings.json are relative patterns (Bash(.venv/bin/python utils/*)), so an
# absolute interpreter path would prompt on every call.

set -euo pipefail

main="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
remove=false
if [[ "${1:-}" == "--remove" ]]; then
    remove=true
    shift
fi

chapter="${1:-}"
if [[ -z "$chapter" ]]; then
    echo "usage: $0 [--remove] <chapter-slug>" >&2
    exit 1
fi

worktree="$(dirname "$main")/mr2-$chapter"
branch="digitize/$chapter"

if $remove; then
    git -C "$main" worktree remove "${@:2}" "$worktree"
    echo "removed $worktree (branch $branch is kept; delete with: git branch -d $branch)"
    exit 0
fi

if [[ -e "$worktree" ]]; then
    echo "error: $worktree already exists" >&2
    exit 1
fi

git -C "$main" worktree add "$worktree" -b "$branch"

ln -s "$main/.venv" "$worktree/.venv"
ln -s "$main/.staging" "$worktree/.staging"
if [[ -f "$main/.claude/settings.local.json" ]]; then
    cp "$main/.claude/settings.local.json" "$worktree/.claude/settings.local.json"
else
    echo "warning: no .claude/settings.local.json in $main;" >&2
    echo "         export MR2_DOCS_MANUAL_PATH in the worktree or the scripts will fail" >&2
fi
mkdir -p "$worktree/logs"

echo
echo "worktree: $worktree"
echo "branch:   $branch"
echo
echo "check it works:"
echo "  cd $worktree && .venv/bin/python utils/manual_map.py --list-remaining --chapter $chapter"
