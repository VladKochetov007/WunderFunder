#!/usr/bin/env bash
set -euo pipefail

# This entrypoint runs inside the prepared receiver image. The host runbook
# supplies the CPU, memory, network, GPU and mounted-input boundaries.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
code_root="${WNN_CODE_ROOT:-$(cd -- "${script_dir}/../../.." && pwd)}"
packet_root="${WNN_PACKET_ROOT:-/packet}"
if [[ ! -d "${packet_root}" ]]; then
  packet_root="${script_dir}"
fi
stage="${WNN_STAGE:-profile}"
profile_sequences="${WNN_PROFILE_SEQUENCES:-2}"
output_dir="${WNN_OUTPUT_DIR:-/outputs}"
zip_path="${WNN_SUBMISSION_ZIP:-${packet_root}/submission.zip}"
inputs_path="${WNN_INPUTS_JSON:-${packet_root}/INPUTS.json}"
recipe_path="${WNN_RECIPE_JSON:-${packet_root}/recipe.json}"
data_path="${WNN_VALID_DATA:-/inputs/valid.parquet}"

fail() {
  echo "receiver: $*" >&2
  exit 2
}

[[ "${stage}" == "profile" || "${stage}" == "full" ]] || fail "WNN_STAGE must be profile or full"
[[ "${profile_sequences}" == "2" || "${profile_sequences}" == "8" ]] || fail "WNN_PROFILE_SEQUENCES must be 2 or 8"
[[ -n "${WNN_ZIP_SHA256:-}" ]] || fail "WNN_ZIP_SHA256 is required"
[[ -n "${WNN_INPUTS_SHA256:-}" ]] || fail "WNN_INPUTS_SHA256 is required"
[[ -n "${WNN_RECIPE_SHA256:-}" ]] || fail "WNN_RECIPE_SHA256 is required"
[[ -r "${zip_path}" && -r "${inputs_path}" && -r "${recipe_path}" && -r "${data_path}" ]] || fail "ZIP, INPUTS, recipe and valid.parquet must be mounted and readable"
[[ -d "${output_dir}" && -w "${output_dir}" ]] || fail "WNN_OUTPUT_DIR must be a writable persistent mount"
[[ -z "${LD_LIBRARY_PATH:-}" ]] || fail "LD_LIBRARY_PATH must be unset for the pinned native ABI"

mkdir -p "${output_dir}"
mkdir -p "${output_dir}/scratch"
export TMPDIR="${output_dir}/scratch"
receipt="${output_dir}/${stage}.json"
[[ ! -e "${receipt}" ]] || fail "refusing to overwrite ${receipt}"

args=(
  --stage "${stage}"
  --zip "${zip_path}"
  --zip-sha256 "${WNN_ZIP_SHA256}"
  --inputs "${inputs_path}"
  --inputs-sha256 "${WNN_INPUTS_SHA256}"
  --recipe "${recipe_path}"
  --recipe-sha256 "${WNN_RECIPE_SHA256}"
  --data "${data_path}"
  --code-root "${code_root}"
  --output "${receipt}"
  --profile-sequences "${profile_sequences}"
  --parity-rows "${WNN_PARITY_ROWS:-2000}"
)
if [[ -n "${WNN_TEST_ROWS:-}" ]]; then
  args+=(--test-rows "${WNN_TEST_ROWS}")
fi
if [[ "${stage}" == "full" ]]; then
  [[ -n "${WNN_PROFILE_RECEIPT:-}" && -n "${WNN_PROFILE_RECEIPT_SHA256:-}" ]] || fail "full stage needs WNN_PROFILE_RECEIPT and WNN_PROFILE_RECEIPT_SHA256"
  args+=(--profile-receipt "${WNN_PROFILE_RECEIPT}" --profile-receipt-sha256 "${WNN_PROFILE_RECEIPT_SHA256}")
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE
exec python -I "${script_dir}/validate_cpu.py" "${args[@]}"
