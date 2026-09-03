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
limit=0
if [[ "${2:-}" == "--limit" ]]; then
    limit="${3:-1}"
fi

if [[ -z "$chapter" ]]; then
    echo "usage: $0 <chapter-slug> [--limit N]" >&2
    exit 1
fi
if [[ ! -f zensical.toml || ! -d utils ]]; then
    echo "error: run this from the root of the chapter's worktree" >&2
    exit 1
fi

mkdir -p logs
done_count=0
failed=()

while IFS= read -r topic; do
    [[ -z "$topic" ]] && continue
    if (( limit > 0 && done_count >= limit )); then
        break
    fi
    slug="$(printf '%s' "$topic" | tr '[:upper:] ' '[:lower:]-' | tr -cd 'a-z0-9-')"
    log="logs/$slug.log"
    echo "=== $topic ==="

    if ! claude -p "/digitize $topic --chapter $chapter" \
            --permission-mode acceptEdits 2>&1 | tee "$log"; then
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
