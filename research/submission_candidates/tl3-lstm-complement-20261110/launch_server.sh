#!/usr/bin/env bash
# Linux SERVER ONLY. This script never starts training or uploads a submission.
set -euo pipefail
stage="${1:-profile}"
[[ "${stage}" == profile || "${stage}" == full ]] || { echo 'Use profile or full' >&2; exit 2; }
delivery="${WNN_DELIVERY_ROOT:?Set absolute directory containing the delivered ZIP and manifests}"
data="${WNN_VALID_DATA:?Set absolute path to the canonical valid.parquet}"
cpu="${WNN_CPU_ID:?Set one currently available logical CPU id}"
image="${WNN_IMAGE_ID:?Set sha256 image id returned by docker image inspect}"
profile_sequences="${WNN_PROFILE_SEQUENCES:-2}"
[[ "${profile_sequences}" == 2 || "${profile_sequences}" == 8 ]] || exit 2
[[ "${image}" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo 'Pin a built image id, not a mutable tag' >&2; exit 2; }
[[ "${delivery}" = /* && "${data}" = /* && "${cpu}" =~ ^[0-9]+$ ]] || exit 2
[[ -f "${data}" && -d "${delivery}" ]] || exit 2
cd "${delivery}"
sha256sum --check SHA256SUMS
out="${delivery}/outputs/${stage}"
mkdir -p "${delivery}/outputs"
mkdir "${out}" # Preserve every earlier run, including incomplete runs.
args=(
  --rm --platform linux/amd64 --network none --cpus 1 --cpuset-cpus "${cpu}"
  --memory 16000000000 --memory-swap 16000000000 --read-only --pids-limit 256
  --cap-drop ALL --security-opt no-new-privileges
  --user "$(id -u):$(id -g)" --entrypoint timeout
  --mount "type=bind,src=${delivery},dst=/packet,readonly"
  --mount "type=bind,src=${data},dst=/inputs/valid.parquet,readonly"
  --mount "type=bind,src=${out},dst=/outputs"
  --env "WNN_STAGE=${stage}" --env "WNN_PROFILE_SEQUENCES=${profile_sequences}"
  --env 'WNN_SUBMISSION_ZIP=/packet/kotiki-tl3-w18-lstm10-cpu-20261110.zip'
  --env 'WNN_ZIP_SHA256=766bab1f3205697819af994acbb54cfe60682651977f393d76bec69608a75e3e'
  --env 'WNN_INPUTS_SHA256=2b751fa835e0c895631a6f167eda6a10880c956bae17d79f581f65f15e5fe190'
  --env 'WNN_RECIPE_SHA256=881c125a78ecf97e8b59986401309b0bd9b2f0b955ee1dcdcb7ae38e82d85ac2'
)
if [[ -n "${WNN_TEST_ROWS:-}" ]]; then
  [[ "${WNN_TEST_ROWS}" =~ ^[1-9][0-9]*$ ]] || exit 2
  args+=(--env "WNN_TEST_ROWS=${WNN_TEST_ROWS}")
fi
limit=1800s
if [[ "${stage}" == full ]]; then
  profile="${delivery}/outputs/profile/profile.json"
  [[ -f "${profile}" ]] || { echo 'Run and inspect profile first' >&2; exit 2; }
  profile_sha="$(sha256sum "${profile}" | cut -d ' ' -f 1)"
  args+=(--env 'WNN_PROFILE_RECEIPT=/packet/outputs/profile/profile.json'
         --env "WNN_PROFILE_RECEIPT_SHA256=${profile_sha}")
  limit=7200s
fi
# No --gpus or host device mounts. Scratch lives on the persistent /outputs bind.
docker run "${args[@]}" "${image}" --signal=TERM --kill-after=30s "${limit}" \
  bash /work/research/submission_candidates/tl3-lstm-complement-20261110/receiver.sh \
  2>&1 | tee "${out}/console.log"
