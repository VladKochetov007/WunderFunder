#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

readonly ROOT=/home/vlad/development/wunder/wnn_connectome_starterpack
readonly PACKET="$ROOT/research/experiment_drafts/online-relationships-20261022"
readonly PROFILE="$PACKET/POSTBOOT_PROFILE_RESULT_R1.json"
readonly PROFILE_EXEC="$PACKET/POSTBOOT_PROFILE_EXECUTION_R1.json"
readonly PROFILE_RECIPE="$PACKET/PROFILE_RECIPE_R3.json"
readonly RECIPE="$PACKET/EXECUTION_RECIPE_R1.json"
readonly SCRATCH=/home/vlad/development/wunder/research_tmp
readonly PYTHON="$ROOT/.venv/bin/python"
readonly PROFILE_SHA=dcbfb56e10ddef133eb9c6c19c5c20068571051416d8d96b8dc7a1c24ba3597e
readonly PROFILE_EXEC_SHA=c635a3e372af9757fac4da534c22ba8cdf35708fb56908f77f940227498be854

fail() { printf 'online delivery preflight failed: %s\n' "$*" >&2; exit 2; }
scratch_path() {
    local resolved
    resolved="$(realpath -m -- "$1")"
    [[ "$resolved" == "$SCRATCH/"* ]] || fail "artifact path outside physical scratch: $resolved"
}
check_sources() {
    local path expected actual
    while IFS=$'\t' read -r path expected; do
        case "$path" in scripts/*|src/*) ;; *) fail "unexpected source path: $path" ;; esac
        actual="$(sha256sum -- "$ROOT/$path" | cut -d ' ' -f 1)"
        [[ "$actual" == "$expected" ]] || fail "profiled source changed: $path"
    done < <(jq -r '.source_sha256 | to_entries[] | [.key,.value] | @tsv' "$PROFILE")
}

for tool in jq sha256sum realpath df find; do command -v "$tool" >/dev/null || fail "missing tool: $tool"; done
[[ -x "$PYTHON" && -f "$PROFILE" && -f "$PROFILE_EXEC" && -f "$PROFILE_RECIPE" && -f "$RECIPE" ]] \
    || fail 'project Python, reviewed profile or admitted recipe is missing'
cd "$ROOT"

[[ "$(sha256sum "$PROFILE" | cut -d ' ' -f 1)" == "$PROFILE_SHA" ]] || fail 'profile result changed'
[[ "$(sha256sum "$PROFILE_EXEC" | cut -d ' ' -f 1)" == "$PROFILE_EXEC_SHA" ]] || fail 'profile execution receipt changed'
[[ "$(sha256sum "$PROFILE_RECIPE" | cut -d ' ' -f 1)" == "$(jq -r .recipe_sha256 "$PROFILE")" ]] \
    || fail 'profile recipe differs from the measured receipt'
jq -e '.sample.fit_sequences_per_fold == 16 and .sample.evaluation_sequences_per_fold == 8 and
    (.head_fits|length) == 12 and .target_scoring_or_quality_selection == "not run"' "$PROFILE" >/dev/null \
    || fail 'profile is not the reviewed cost-only R3 sample'
jq -e '.status == "COMPLETE_ENGINEERING_PROFILE_ONLY" and .source_code_last_change == "af952697f" and
    .scientific_fits_admitted == 0 and .returncode == 0' "$PROFILE_EXEC" >/dev/null \
    || fail 'profile execution is not complete or is scientific'
check_sources

# Only admission/status and execution metadata may differ from the profiled recipe.
jq -e --slurpfile base "$PROFILE_RECIPE" '
    .status == "admitted_for_execution" and .admitted_new_fits == 12 and
    (del(.status,.admitted_new_fits,.execution_actual_cpu_backend,.execution_metadata) ==
     ($base[0] | del(.status,.admitted_new_fits,.execution_actual_cpu_backend,.execution_metadata)))
' "$RECIPE" >/dev/null || fail 'execution recipe changes the profiled panel or lacks full-panel admission'
jq -e '(.folds|length)==2 and ([.folds[].parent_cache]|unique|length)==2 and
    (.cells|length)==12 and ([.cells[].output]|unique|length)==12' "$RECIPE" >/dev/null \
    || fail 'recipe does not bind exactly two caches and twelve unique outputs'

while IFS= read -r dir; do
    scratch_path "$dir"
    [[ ! -d "$dir" || -z "$(find "$dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] \
        || fail "parent cache path is not empty: $dir"
done < <(jq -r '.folds[].parent_cache' "$RECIPE")
while IFS= read -r dir; do
    scratch_path "$dir"
    [[ ! -e "$dir" ]] || fail "readout output already exists: $dir"
done < <(jq -r '.cells[].output' "$RECIPE")
free="$(df -PB1 --output=avail "$SCRATCH" | tail -n 1 | tr -d '[:space:]')"
[[ "$free" =~ ^[0-9]+$ ]] && (( free >= 40000000000 )) || fail 'less than 40 GB physical scratch is available'

# The outer thermal guard owns the shared gpu.lock and enforces CPU/RAM/swap/temperature/time limits.
recipe_sha="$(sha256sum "$RECIPE" | cut -d ' ' -f 1)"
printf 'online delivery phase=prepare_cache recipe_sha256=%s\n' "$recipe_sha"
"$PYTHON" -u "$ROOT/scripts/online_relationships.py" --root "$ROOT" prepare-cache --recipe "$RECIPE" --fold all --device cpu
printf 'online delivery phase=fit_panel recipe_sha256=%s\n' "$recipe_sha"
"$PYTHON" -u "$ROOT/scripts/online_relationships.py" --root "$ROOT" fit-panel --recipe "$RECIPE" --device cpu

[[ "$(sha256sum "$RECIPE" | cut -d ' ' -f 1)" == "$recipe_sha" ]] || fail 'execution recipe changed during delivery'
while IFS= read -r dir; do
    for file in fit_sequences.npy fit_sequences.json evaluation_sequences.npy evaluation_sequences.json \
        evaluation_predictions.npy evaluation_predictions.json relationship_normalizer_lag1.json relationship_normalizer_lag16.json; do
        [[ -s "$dir/$file" ]] || fail "cache artifact missing: $dir/$file"
    done
done < <(jq -r '.folds[].parent_cache' "$RECIPE")
while IFS= read -r dir; do
    for file in model_ep2.pt prediction.npy prediction.json provenance.json; do
        [[ -s "$dir/$file" ]] || fail "readout artifact missing: $dir/$file"
    done
done < <(jq -r '.cells[].output' "$RECIPE")
printf 'online delivery complete: two parent caches and all twelve readout endpoints exist; grading is separate\n'
