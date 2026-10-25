#!/usr/bin/env bash
# Receiver-only metadata/data preparation. No tests, GPU profiles or fits.
set -euo pipefail

handoff=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
packet=$(cd -- "$handoff/../high_compute_r4" && pwd)
cd -- "$packet/../../../.."
[[ $(uname -s) == Linux ]] || { echo 'Prepare on the receiving Linux host' >&2; exit 2; }
: "${WNN_EXPECTED_COMMIT:?pin the reviewed full handoff commit SHA}"
: "${WNN_DATA_ROOT:?set the existing data root}"
: "${WNN_SPLIT_F0:?set the canonical F0 manifest}"
: "${WNN_SPLIT_F2:?set the canonical F2 manifest}"
: "${WNN_INPUTS_ROOT:?set a new external inner-split directory}"
: "${WNN_CAMPAIGN_ROOT:?set a new external R4 campaign directory}"
[[ $WNN_EXPECTED_COMMIT =~ ^[0-9a-f]{40}$ ]] || { echo 'A full commit SHA is required' >&2; exit 2; }
[[ $(git rev-parse HEAD) == "$WNN_EXPECTED_COMMIT" ]] || { echo 'Source commit mismatch' >&2; exit 2; }
[[ -z $(git status --porcelain -- src scripts tests "$packet/.." pyproject.toml uv.lock) ]] || {
  echo 'Runtime, tests, packet or dependencies are dirty' >&2; exit 2;
}
[[ -d $WNN_DATA_ROOT ]] || { echo 'Missing data root' >&2; exit 2; }
for target in "$WNN_INPUTS_ROOT" "$WNN_CAMPAIGN_ROOT"; do
  [[ $target == /* && ! -e $target && ! -L $target ]] || {
    echo 'Use fresh absolute output paths; retain any previous attempt' >&2; exit 2;
  }
  resolved=$(realpath -m -- "$target")
  case "$resolved/" in
    "$PWD/"*|/tmp/*|/dev/shm/*|*/pr32-teachers-*/*|*/pr32-source-*/*)
      echo 'Output must be outside source/live paths; choose physical-disk storage' >&2; exit 2 ;;
  esac
done
[[ $(realpath -m -- "$WNN_INPUTS_ROOT") != $(realpath -m -- "$WNN_CAMPAIGN_ROOT") ]] || {
  echo 'Inner-split and campaign roots must differ' >&2; exit 2;
}
# These are the canonical outer manifests recorded by the current PR40 cohort.
# Changed bytes need a separately reviewed data/split binding, not a bypass.
[[ $(sha256sum -- "$WNN_SPLIT_F0" | cut -d ' ' -f 1) == 15851c8055806566ff326793073cf24706b1c684c5bccd623302c999d31be041 ]] || {
  echo 'Canonical F0 manifest hash mismatch' >&2; exit 2;
}
[[ $(sha256sum -- "$WNN_SPLIT_F2" | cut -d ' ' -f 1) == 9fff6617d7f701735c2edee57945d33d6e3cce3573197cd7980b927b8b504fa4 ]] || {
  echo 'Canonical F2 manifest hash mismatch' >&2; exit 2;
}
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -- "$WNN_INPUTS_ROOT"
uv run --no-sync python scripts/prepare_sequence_inner_split.py \
  --parent-manifest "$WNN_SPLIT_F0" --data-root "$WNN_DATA_ROOT" \
  --seed 41001 --evaluation-fraction 0.2 --output "$WNN_INPUTS_ROOT/inner-f0.json"
uv run --no-sync python scripts/prepare_sequence_inner_split.py \
  --parent-manifest "$WNN_SPLIT_F2" --data-root "$WNN_DATA_ROOT" \
  --seed 41002 --evaluation-fraction 0.2 --output "$WNN_INPUTS_ROOT/inner-f2.json"
uv run --no-sync python "$packet/campaign.py" init \
  --campaign-root "$WNN_CAMPAIGN_ROOT" --data-root "$WNN_DATA_ROOT" \
  --outer-f0 "$WNN_SPLIT_F0" --outer-f2 "$WNN_SPLIT_F2" \
  --inner-f0 "$WNN_INPUTS_ROOT/inner-f0.json" --inner-f2 "$WNN_INPUTS_ROOT/inner-f2.json"
uv run --no-sync python "$handoff/run.py" prepare --campaign-root "$WNN_CAMPAIGN_ROOT"
printf '%s\n' 'Four fresh 1.54B fits prepared with a component-disjoint train/validation audit. No fits launched.'
