# Portable xteam_h4 CPU policy panel

This packet runs the original xteam_h4 Python callback against the frozen Q8 policy packages A/B/C/D on a CPU-only benchmark host. It separates a small correctness smoke from a full timing panel and does not train, score, rebuild, or upload a model. The packages retain the original ensemble weights, member order, coefficients, preprocessing, GRU architecture and Q8 codes/scales; the exact package and archive hashes are in [the A/B/C/D package binding](PACKAGE_BINDING_R1.json).

The binding generator checks each archive's SHA-256, byte length, CRC, safe unique member names, per-member hashes and extracted-file hashes. It also checks that `members.json` and all four weight archives match across policies. The source archive is the original `xteam_h4.zip` (`f1635d14c0e1b47848b5caeb2ec4ad9362a85800f1b2e2df80702127d3867b27`); B/C/D archives are the already-built packages from [the October 14 package manifest](../inference_policy_r1/PACKAGE_MANIFEST_R1.json). No native build is part of this packet.

The label-free runtime fixture is the eight-sequence F0 seed-3 fitting panel produced by [`prepare_vast_a_cpu_probe_fixture.py`](../../../campaign_2026-10-15/vast_a_cpu_probe_r1/prepare_vast_a_cpu_probe_fixture.py), with its [frozen manifest](../../../campaign_2026-10-15/vast_a_cpu_probe_r1/CPU_TIMING_FIXTURE_R1.json) and payloads at `artifacts/vast_a_cpu_probe_r1/`. The committed manifest SHA-256 is `941b656d8408be6f5c8f5816c1e32f3c8b24e59040bb4ed84bb895078a9f9f05`. The consumer verifies the whole compact feature/request payloads and per-sequence hashes. Source identity binds the fold-0 Parquet footer hash, file size and mtime, not a fresh hash of the 29 GB source body. The request vector asks for predictions on every row from 99 onward; it is a synthetic runtime request schedule, not a competition scoring mask. No labels, official mask, teachers, caches or propensity values are present.

Each callback creates the same four-field `DataPoint` contract as pinned [`utils.py`](../../../../utils.py): `seq_ix`, `step_in_seq`, `need_prediction`, and `state`. The runner avoids importing the scorer's PyArrow-dependent module. It imports each package's unchanged `solution.PredictionModel`, checks the selected policy and loaded native-library path/hash, then calls `.predict` once for every row of the complete raw112 sequence. Each replay begins after a 99-row sentinel sequence switch; sentinel reset time is reported separately and excluded from callback time. A/B/C/D are loaded in separate persistent Python 3.11 workers, all pinned to one logical CPU. CPU quota is recorded, not claimed as enforced by the worker.

Run the admitted compatibility smoke with Python 3.11 and NumPy. It hashes/maps the fixture and performs two complete, reset-separated callbacks each for A and B on one source-qualified sequence. It checks deterministic reset replay and reports bitwise A/B equality or a numerical delta; this smoke does not rank timing or establish panel-wide parity.

```sh
WNN_SLICE=wnn-campaign.slice WNN_CPU_QUOTA=100% WNN_THREADS=1 \
WNN_MEM_MAX=1500000000 WNN_RUNTIME_MAX=180s \
scripts/limited /path/to/python3.11 scripts/benchmark_cpu_policy_panel_r1.py \
  --smoke-pair \
  --package-root /path/to/staged/A-B-C-D \
  --package-binding research/campaign_2026-10-14/cpu_student_frontier_r1/portable_cpu_benchmark_r1/PACKAGE_BINDING_R1.json \
  --fixture-dir artifacts/vast_a_cpu_probe_r1 \
  --fixture-manifest research/campaign_2026-10-15/vast_a_cpu_probe_r1/CPU_TIMING_FIXTURE_R1.json \
  --output /path/to/new-smoke-receipt
```

The full panel is prepared but has not been run. It uses a five-callback A/B/C/D/A pilot, then the same eight-sequence panel in every formal block. For each sequence A brackets the candidate trio; the B/C/D order traverses all six permutations exactly once in each six-block cycle. A minimum of six blocks per candidate is mandatory. After that, the panel may stop when all wall-time paired intervals reach the requested half-width, or continue to a maximum of 12 blocks. A full block has 40 timed complete-sequence callbacks. A/B/C/D callback order and sequence order are frozen by seed `20261015`. All blocks and adaptive intervals are exploratory; they do not confirm a selected winner. Any later winner claim needs untouched confirmation blocks after the adaptive panel stops.

```sh
scripts/benchmark_cpu_policy_panel_r1.py \
  --package-root /path/to/staged/A-B-C-D \
  --package-binding research/campaign_2026-10-14/cpu_student_frontier_r1/portable_cpu_benchmark_r1/PACKAGE_BINDING_R1.json \
  --fixture-dir artifacts/vast_a_cpu_probe_r1 \
  --fixture-manifest research/campaign_2026-10-15/vast_a_cpu_probe_r1/CPU_TIMING_FIXTURE_R1.json \
  --output /path/to/new-panel-receipt \
  --seed 20261015 --min-blocks 6 --max-blocks 12 \
  --precision-halfwidth-pct 1 --time-budget-s 1000
```

The output reports callback wall and worker process time, per-sequence repeats, between-sequence means, block-paired log-ratio intervals, fixture/package verification, constructor/weight packing, process spawn-to-ready, cgroup counters, CPU identity and every reset/parity result. Cold start, construction, fixture hashing and sentinel reset are separate from callback time. Intervals are descriptive and conditional on one host; sequential stopping is not adjusted, and a near-tie stays unresolved. No runtime score or public result is computed.

Four pure-protocol unit tests pass for block balance, minimum budget admission, paired intervals and the separate within-/between-sequence summaries. A local compatibility smoke receipt will be added separately after its bounded callback run. The target AMD EPYC 9224 timing panel remains pending root-owned CPU-host availability; no Vast instance, payment, GPU or remote action was initiated.

## Files

- [`benchmark_cpu_policy_panel_r1.py`](../../../../scripts/benchmark_cpu_policy_panel_r1.py): fixture/package verification, one-sequence A/B smoke and paired A/B/C/D timing worker.
- [`build_cpu_policy_package_binding_r1.py`](../../../../scripts/build_cpu_policy_package_binding_r1.py): CRC/member/hash verification and static package binding generator.
- [`test_cpu_policy_panel_r1.py`](../../../../tests/test_cpu_policy_panel_r1.py): deterministic protocol and summary checks.
- [`BENCHMARK_PACKET_R1.json`](BENCHMARK_PACKET_R1.json): machine-readable commands, panel order, boundaries and current execution status.
- [`PACKAGE_BINDING_R1.json`](PACKAGE_BINDING_R1.json): immutable archive and extracted-package identities.
