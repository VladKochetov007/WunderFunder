#!/usr/bin/env bash
# commit.sh "message" path...   (adds listed paths, commits only if something is staged)
set -euo pipefail
msg=$1; shift
git add -A -- "$@" 2>/dev/null || true
if git diff --cached --quiet; then echo "SKIP (nothing staged): ${msg%%$'\n'*}"; exit 0; fi
git commit -q -F - <<MSG
$msg

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
echo "OK $(git log -1 --format='%h %s')"
