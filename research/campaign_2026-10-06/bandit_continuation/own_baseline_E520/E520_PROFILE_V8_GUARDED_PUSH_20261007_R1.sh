#!/usr/bin/env bash
set -euo pipefail

repo=/home/vlad/development/wunder/wnn_connectome_starterpack
cd "$repo"
packet=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/packet_v8
manifest=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/E520_PROFILE_V8_PACKET_SHA256SUMS_20261007_R1.txt
controller=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/E520_PROFILE_V8_CONTROLLER_ADMISSION_20261007_R1.json
assembly=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/E520_PROFILE_V8_PACKET_ASSEMBLY_20261007_R1.json
source_review=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/pre_cuda_gpu_fallback_v8_candidate_v1/E520_V8_INDEPENDENT_REVIEW_20261007_R1.json
transport_review=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/E520_PROFILE_V8_PACKET_TRANSPORT_INDEPENDENT_REVIEW_20261007_R1.json
registration=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/E520_PROFILE_V8_PROVIDER_TARGET_REGISTRATION_20261007_R1.json
targets=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/cloud_preflight/provider_targets.json
prepush=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/E520_PROFILE_V8_PREPUSH_ADMISSION_20261007_R1.json
script_path=/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/E520_PROFILE_V8_GUARDED_PUSH_20261007_R1.sh

if ! printf '%s  %s\n' 543c8b07b859a4761a013ad99b3854f4b11aa19e0374bf5d011de537d19d5147 "$manifest" | sha256sum --check --status; then
  echo "packet manifest hash mismatch; refusing push" >&2
  exit 1
fi
if ! (cd "$packet" && sha256sum --check --status "$manifest"); then
  echo "packet member hash mismatch; refusing push" >&2
  exit 1
fi
if ! printf '%s  %s\n' fbc45dc2a6c29870620bb1a5040d709a5152eeee5b0e45c55ceb922039676417 "$controller" | sha256sum --check --status; then
  echo "controller build admission hash mismatch; refusing push" >&2
  exit 1
fi
if ! printf '%s  %s\n' e1986014729abafc45df8f019aa06d9b40923f9eeef8b831c26d54699e383a81 "$assembly" | sha256sum --check --status; then
  echo "packet assembly receipt hash mismatch; refusing push" >&2
  exit 1
fi
if ! printf '%s  %s\n' faddd1dc3ccfe7aed085c7e926280439b9817ea672a92cc6e1a1cb9f0bb37f9d "$source_review" | sha256sum --check --status; then
  echo "source review hash mismatch; refusing push" >&2
  exit 1
fi
if ! printf '%s  %s\n' 91a2254cf3a9afd0ab449a24a691b657f8c27c1f65ce06a2854428257ec0d834 "$transport_review" | sha256sum --check --status; then
  echo "transport review hash mismatch; refusing push" >&2
  exit 1
fi
if ! printf '%s  %s\n' be19acee413c6673a80fb3ff21dd14e29bd5cea9dd1614899dc033df84f21fba "$registration" | sha256sum --check --status; then
  echo "provider registration receipt hash mismatch; refusing push" >&2
  exit 1
fi
if ! printf '%s  %s\n' 4744b0e8b613b0585422f9e30bc33b9173c93aa2dff61ed1e5fedaa83d85bb20 "$targets" | sha256sum --check --status; then
  echo "provider target registry changed after admission; refusing push" >&2
  exit 1
fi

WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=1000000000 WNN_RUNTIME_MAX=120s \
scripts/limited .venv/bin/python -c 'import hashlib,json,pathlib,sys; s=pathlib.Path(sys.argv[1]); a=json.loads(pathlib.Path(sys.argv[2]).read_text()); h=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest(); assert a.get("guarded_push_script_sha256")==h(s), "push script hash mismatch"; assert a.get("status")=="approved" and a.get("provider_push_now_authorized") is True; assert a.get("fit_or_score_admitted") is False; assert a.get("kernel_id")=="vladkochetov07/e520-profile-v8-gpu-observation-fallback"; assert a.get("controller_admission_sha256")==h(sys.argv[3]); assert a.get("packet_assembly_sha256")==h(sys.argv[4]); assert a.get("packet_manifest_sha256")==h(sys.argv[5]); assert a.get("transport_review_sha256")==h(sys.argv[6]); assert a.get("provider_target_registration_sha256")==h(sys.argv[7]); assert a.get("quota_admission",{}).get("e520_remaining_after_attempt_reservation_seconds",-1)>=0; assert a.get("quota_admission",{}).get("campaign_headroom_after_attempt_and_conditional_fit_seconds",-1)>0; assert a.get("quota_admission",{}).get("account_headroom_after_reserve_attempt_and_conditional_fit_seconds",-1)>0' "$script_path" "$prepush" "$controller" "$assembly" "$manifest" "$transport_review" "$registration"

export WNN_SLICE=wnn-campaign.slice
export WNN_CPU_QUOTA=100%
export WNN_THREADS=1
export WNN_MEM_MAX=1000000000
export WNN_RUNTIME_MAX=120s
exec scripts/limited timeout --signal=TERM --kill-after=5s 100s kaggle kernels push -p "$packet" -t 1500
