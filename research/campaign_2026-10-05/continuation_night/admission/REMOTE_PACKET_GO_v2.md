# Independent remote packet GO, v2/v3

Disposition: **GO for root's per-job reservation and launch review of the three
new packets listed below.** This is a source, guard and packet-integrity review;
it did not upload a kernel or start a provider job. The controller must still
refresh free quota, active jobs and attempt counts and bind a separate reservation
to one exact packet before any push or run. Old packet directories remain
preserved; this review covers only the v2 GPU and v3 TPU packet IDs below.

## Resource guard review

The revised cgroup-v1 sampler walks the visible memory hierarchy from leaf
through mount root. It requires readable, valid limit and usage values at each
level, applies every level to effective capacity/headroom, and fails closed on
missing or malformed ancestors. This handles an intermediate parent limit even
when the leaf advertises the v1 unlimited sentinel. OOM sampling now extracts
`oom_kill` from v1 `memory.oom_control` as well as v2 counters from
`memory.events`; the supervisor compares keyed counters with the prelaunch
baseline. Process supervision retains the subreaper, adopted-child inventory and
cleanup checks.

I independently added fixtures for v1 intermediate headroom, v1 `oom_kill`
visibility, and missing or malformed ancestor rejection. I reran both the
transport suite and the independent remote fixtures under one CPU, two GiB RAM,
GPU hidden and plugin autoload disabled:

```bash
env WNN_CPU_QUOTA=100% WNN_THREADS=1 WNN_MEM_MAX=2G CUDA_VISIBLE_DEVICES= \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=research/campaign_2026-10-05/continuation_night/width:research/campaign_2026-10-05/continuation_night/kaggle:src \
  scripts/limited .venv/bin/python -m pytest \
  research/campaign_2026-10-05/continuation_night/width/test_transport.py \
  research/campaign_2026-10-05/continuation_night/admission/test_remote_independent.py -q
```

Result: **18 passed in 30.28 seconds**. Coverage includes complete teacher,
propensity and mask tensor byte parity; cgroup-v2 and cgroup-v1 ancestor
headroom; absent/malformed v1 ancestor rejection; v1 OOM counter extraction;
root-exit child adoption/reaping; timeout cleanup of a nested process session;
and packet source derivation checks. No science or CUDA work ran in these
fixtures.

## Packet integrity and privacy

I parsed each current kernel without executing it, decompressed its embedded
sources, and verified each SHA256 against the embedded hash table. Each packet's
kernel bytes, metadata bytes, review receipt, immutable input manifest and
original source payload match the current registry. All three metadata records
are private, disable internet, and contain no competition or kernel attachments.
The packets retain one scientific worker per notebook, the host 70% target / 85%
stop, and explicit root reservation and free-quota requirements.

| Prepared packet | Kernel SHA256 | Review receipt SHA256 |
|---|---|---|
| `vladkochetov07/wnn-night-w256-serial-20261005-v2` | `f56ae9471b42bffeb32186d2e75fe259d8593bcfefeb564cb8d5471b2949ce81` | `de4987c5c698444703baa38355cec24ddfc02ca9c3b37f7298a5ee3ddf21f32f` |
| `vladkochetov07/wnn-night-w384-serial-20261005-v2` | `62f8ced69bb79f70b5a801f57f500a74f7d362a308248b1b4b284083341aaf7a` | `3d8f6bd1e47e4aa52fe181b1a75b7c99bbd94951d57a86e029eb769d83e0ea49` |
| `vladkochetov07/wnn-night-e463-tpu-20261005-v3` | `99ad1fd3aed76f6d28810f590f99a463da51d7bbd8cffcbacc3361e9dbe23994` | `ecbe9df24d54d808a70cc517a5aa532431e90d3cc08d1e7c62cd70ceb8f90cfa` |

The GPU packets bind manifest `4518e583e9eab4bcaa5c5eb499958e9329748191006e68cd89a101d12cee1958`
and original payload `498c0a5d1105a9fbbe34d02fffb7d4a6c12cd5d2f6b10f8ef647b7bac03bb90a`.
The TPU packet binds manifest `851898475dfa701f2089ec76cfe9592538aa2e7f9b70ebcb68c0a614e332d6fb`
and original payload `305040488efdccc5eaab1446ee4d8a522fcb3bf60737917da0f75d29ed1252db`.
The controlling `PACKETS.json` SHA256 is
`3bc093679d83be2328f45402c6b80b6e19df72342c5dcab8c5deb02e034c828c`.

Reviewed implementation identities:

| File | SHA256 |
|---|---|
| `kaggle/host_guard.py` | `8ca8a97186b808c47289f3c379c923f63ca3fbe0e93bb6ea91ba6f43536ebdfa` |
| `kaggle/build_packets.py` | `472be9637e59525366d26232dd2e4017e896e38b2e932ac4e87ee6bcf3ff7143` |
| `width/test_transport.py` | `bb77920dfac1b51e14ca4765056ab914fcfddb1e3737b80d623bb42b765758b0` |
| `admission/test_remote_independent.py` | `7220be7a768849dfe428ced30d5058328e531578f12251f215553441ea22b485` |

Remote admission remains a systems preflight only. It does not authorize E465
production training, another recovery attempt, paid compute beyond the exact
root-reserved packet attempts, or a scientific claim. The root controller owns
quota, active-job and budget admission and must bind each launch to one of the
exact packet hashes above. The reviewed cgroup-v1 ancestor and OOM-counter fixes
are included in these packet versions.
