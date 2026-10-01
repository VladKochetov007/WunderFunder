#!/usr/bin/env bash
# N1 screen: joint vs solo extra head FT for a fixed member set on clean fold K.
#   joint_heads.sh FOLD TAG MEMBER@EP [MEMBER@EP ...]   (members may use {fold})
set -euo pipefail
cd "$(dirname "$0")/../../.."
fold=$1 tag=$2; shift 2
members=()
for m in "$@"; do members+=("${m//\{fold\}/$fold}"); done
for mode in joint solo; do
  .venv/bin/python scripts/joint_head_ft.py "c20261001_${tag}_${mode}_f${fold}" --fold "$fold" --mode "$mode" \
    --members "${members[@]}"
done
