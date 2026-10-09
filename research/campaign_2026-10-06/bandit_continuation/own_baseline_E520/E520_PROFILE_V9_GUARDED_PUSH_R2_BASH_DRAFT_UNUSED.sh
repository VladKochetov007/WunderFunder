#!/usr/bin/env bash
set -euo pipefail

repo=/home/vlad/development/wunder/wnn_connectome_starterpack
cd "$repo"
base="$repo/research/campaign_2026-10-06/bandit_continuation/own_baseline_E520"
packet="$base/packet_v9"
manifest="$base/E520_PROFILE_V9_PACKET_SHA256SUMS_20261007_R1.txt"
controller="$base/E520_PROFILE_V9_CONTROLLER_BUILDER_ADMISSION_20261007_R1.json"
assembly="$base/E520_PROFILE_V9_PACKET_ASSEMBLY_20261007_R1.json"
source_review="$base/E520_PROFILE_V9_INDEPENDENT_SOURCE_REVIEW_20261007_R1.json"
transport_review="$base/E520_PROFILE_V9_PACKET_TRANSPORT_INDEPENDENT_REVIEW_20261007_R1.json"
registration="$base/E520_PROFILE_V9_PROVIDER_TARGET_REGISTRATION_20261007_R1.json"
allocation="$base/E520_V9_ROOT_CONDITIONAL_REPAIR_ALLOCATION_20261007_V1.json"
targets="$repo/research/campaign_2026-10-06/cloud_preflight/provider_targets.json"
prepush="$base/E520_PROFILE_V9_PREPUSH_ADMISSION_20261007_R2.json"
script_path="$base/E520_PROFILE_V9_GUARDED_PUSH_20261007_R2.sh"
slug=vladkochetov07/e520-profile-v9-memory-domains
kernel_list="$base/E520_V9_KERNEL_LIST_IMMEDIATE_PREPUSH_20261007_R2.txt"
kernel_list_err="$base/E520_V9_KERNEL_LIST_IMMEDIATE_PREPUSH_20261007_R2.stderr.txt"
attempt_marker="$base/E520_PROFILE_V9_PUSH_ATTEMPT_STARTED_20261007_R2.json"

if [[ $# -ne 1 ]]; then
  echo "expected the exact pre-push admission SHA-256" >&2
  exit 2
fi
admission_sha="$1"
if ! printf '%s  %s\n' "$admission_sha" "$prepush" | sha256sum --check --status; then
  echo "pre-push admission hash mismatch; refusing push" >&2
  exit 1
fi

verify_sha() {
  local expected="$1"
  local file="$2"
  if ! printf '%s  %s\n' "$expected" "$file" | sha256sum --check --status; then
    echo "SHA-256 mismatch for $file; refusing push" >&2
    exit 1
  fi
}

verify_sha a019fb208c95b9a986c00077c7201d2b70c126e7630b5a02b395a762897359ad "$manifest"
if ! (cd "$packet" && sha256sum --check --status "$manifest"); then
  echo "packet member hash mismatch; refusing push" >&2
  exit 1
fi
verify_sha 14f1c9d3500f5ebaeb4281f34451dc5f4b313e02125dd24b78a7893f479de436 "$controller"
verify_sha 0817674b77d35b2c31dbd708ad86ea8afa047880a90dddd386da2e2035f2de07 "$assembly"
verify_sha b8c2d3f9dbb481a6009c4f8bd6e4ae2d5b7e40ea2327d1e0f44baa22e5d2f1f7 "$source_review"
verify_sha 212606d0ccc20a002d291dd9e9c7baa0026e91721b3222c31066b1655102ed40 "$transport_review"
verify_sha e9ab52462f8e1520592454679e890940d2ec841debf61f0c9da3bae703a6112d "$registration"
verify_sha 67302c079b6d04e343eb0676717bbd11d23ccd191abe87c12f21a93c21a83784 "$allocation"
verify_sha 30503ecd19f1ce095210237f6755f6b7ffcba5b6131d662e343aee9d51eecdac "$targets"

.venv/bin/python -c 'import hashlib,json,pathlib,sys; script=pathlib.Path(sys.argv[1]); admission=pathlib.Path(sys.argv[2]); controller=pathlib.Path(sys.argv[3]); assembly=pathlib.Path(sys.argv[4]); manifest=pathlib.Path(sys.argv[5]); source=pathlib.Path(sys.argv[6]); transport=pathlib.Path(sys.argv[7]); registration=pathlib.Path(sys.argv[8]); allocation=pathlib.Path(sys.argv[9]); h=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest(); a=json.loads(admission.read_text()); t=json.loads(transport.read_text()); assert a.get("guarded_push_script_sha256")==h(script),"push script hash mismatch"; assert a.get("status")=="approved" and a.get("provider_push_now_authorized") is True; assert a.get("fit_or_score_admitted") is False; assert a.get("kernel_id")=="vladkochetov07/e520-profile-v9-memory-domains"; assert a.get("controller_admission_sha256")==h(controller); assert a.get("packet_assembly_sha256")==h(assembly); assert a.get("packet_manifest_sha256")==h(manifest); assert a.get("source_review_sha256")==h(source); assert a.get("transport_review_sha256")==h(transport); assert a.get("provider_target_registration_sha256")==h(registration); assert a.get("root_allocation_sha256")==h(allocation); assert t.get("decision")=="PASS" and t.get("freeze",{}).get("assembly_receipt_sha256")==h(assembly); q=a.get("quota_admission",{}); assert q.get("e520_remaining_after_attempt_reservation_seconds",-1)>=0; assert q.get("campaign_headroom_after_attempt_and_conditional_fit_seconds",-1)>0; assert q.get("account_headroom_after_reserve_attempt_and_conditional_fit_seconds",-1)>0' "$script_path" "$prepush" "$controller" "$assembly" "$manifest" "$source_review" "$transport_review" "$registration" "$allocation"

if ! kaggle kernels list --mine --search e520-profile-v9-memory-domains --format json > "$kernel_list" 2> "$kernel_list_err"; then
  echo "exact-slug absence query failed; refusing push" >&2
  exit 1
fi
if [[ "$(cat "$kernel_list")" != "Not found" || -s "$kernel_list_err" ]]; then
  echo "exact private slug is no longer absent; refusing push" >&2
  exit 1
fi

printf '{"attempt_id":"E520_PROFILE_GPU_MEMORY_DOMAIN_V9","status":"provider_push_invoked","invoked_at_utc":"%s","kernel_id":"%s","prepush_admission_sha256":"%s","guarded_push_script_sha256":"%s","automatic_retry":false}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$slug" "$admission_sha" "$(sha256sum "$script_path" | cut -d' ' -f1)" > "$attempt_marker"

exec timeout --signal=TERM --kill-after=5s 100s kaggle kernels push -p "$packet" -t 1700
