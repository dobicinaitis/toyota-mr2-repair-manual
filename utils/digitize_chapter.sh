#!/usr/bin/env bash
# Digitizes every remaining topic of one chapter, one headless Claude run per topic.
#
# Usage:
#     utils/worktree.sh turbocharger-system
#     cd ../mr2-turbocharger-system
#     utils/digitize_chapter.sh turbocharger-system            # all remaining topics
#     utils/digitize_chapter.sh turbocharger-system --limit 1  # just the next one
#
# Run this from inside the chapter's worktree: every script defaults to paths
# relative to the current directory, so from anywhere else it would lint and
# rewrite the wrong tree.
#
# One run per topic rather than one per chapter: a topic costs two image reads per
# page over up to 58 pages, and a fresh context window per topic is worth more than
# carrying the previous topic along. Each finished topic is committed on its own, so
# an interrupted chapter resumes where it stopped (--list-remaining skips what exists).
#
# The site-wide steps -- resolve_refs.py, build_glossary.py, sync_nav.py and the
# zensical build -- are deliberately NOT run here. resolve_refs.py rewrites pages
# across the whole site as new anchors appear, so two worktrees running it produce
# conflicting edits to the same committed files. They belong to the merge pass in
# CLAUDE.md instead.

set -uo pipefail

chapter="${1:-}"
shift || true
limit=0
# Sonnet by default. Benchmarked against the digitized Troubleshooting, Engine
# tune-up and Compression check pages: same page anchors, same illustrations,
# and an identical set of values, including the shim selection charts. Pass
# --model for a topic worth spending more on.
model=(--model sonnet)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --limit) limit="${2:-1}"; shift 2 ;;
        --model) model=(--model "${2:?--model needs a value}"); shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$chapter" ]]; then
    echo "usage: $0 <chapter-slug> [--limit N] [--model NAME]   (default model: sonnet)" >&2
    exit 1
fi
if [[ ! -f zensical.toml || ! -d utils ]]; then
    echo "error: run this from the root of the chapter's worktree" >&2
    exit 1
fi

# Stopping this script must stop the run it is waiting on. Without this, killing
# the loop leaves the claude process orphaned: it keeps working in the worktree,
# writes files nobody is expecting, and the next run skips those topics because
# their markdown now exists.
trap 'pkill -P $$ 2>/dev/null' EXIT INT TERM

mkdir -p logs
done_count=0
failed=()

# .staging is a symlink to the main checkout, which is outside this worktree.
# The scripts follow it fine, but the agent's own file reads are refused at the
# filesystem boundary before the Read(.staging/**) permission rule is consulted,
# so the staged OCR text and page overlays have to be allowed explicitly. Only
# the staging directory is added, not the whole main checkout.
staging="$(readlink -f .staging)"
add_dir=()
if [[ -n "$staging" && "$staging" != "$PWD/.staging" ]]; then
    add_dir=(--add-dir "$staging")
fi

while IFS= read -r topic; do
    [[ -z "$topic" ]] && continue
    if (( limit > 0 && done_count >= limit )); then
        break
    fi
    slug="$(printf '%s' "$topic" | tr '[:upper:] ' '[:lower:]-' | tr -cd 'a-z0-9-')"
    log="logs/$slug.log"
    echo "=== $topic ==="

    # </dev/null matters: claude reads stdin, and this loop is being fed the topic
    # list on stdin. Without it the first run swallows the remaining topics and the
    # loop exits after one iteration, whatever --limit says.
    if ! claude -p "/digitize $topic --chapter $chapter" \
            "${add_dir[@]}" "${model[@]}" --permission-mode acceptEdits < /dev/null 2>&1 | tee "$log"; then
        echo "FAILED: $topic (see $log)" >&2
        failed+=("$topic")
        continue
    fi

    # a topic counts as done only if its page actually appeared
    path="$(.venv/bin/python utils/manual_map.py --topic "$topic" --chapter "$chapter" --json |
            .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["path"])')"
    if [[ ! -f "docs/$path" ]]; then
        echo "FAILED: $topic produced no docs/$path (see $log)" >&2
        failed+=("$topic")
        continue
    fi

    git add -A
    git commit -q -m "Added the $topic section" && echo "committed $topic"
    done_count=$((done_count + 1))
done < <(.venv/bin/python utils/manual_map.py --list-remaining --chapter "$chapter")

echo
echo "$done_count topic(s) digitized in $chapter"
if (( ${#failed[@]} )); then
    printf 'needs a human: %s\n' "${failed[@]}" >&2
    exit 1
fi
