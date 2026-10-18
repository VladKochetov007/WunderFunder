# W18 checkpoint metadata preparation R1

**Prepared only; no remote launch or checkpoint read.** This packet narrows a follow-up metadata pass to six newly identified W18 teacher endpoints whose exact logical paths and stat sizes were recorded by the bounded probe in [the W18 history trace](../tl3_w18_history_trace_r1/README.md). It does not copy checkpoint bytes, inspect tensors, read prediction or propensity arrays, train, infer, or alter source assets.

The source observation was recorded at `2026-10-18T17:49:55Z`. It found these six endpoints under `/home/grinchenkoeg/WunderFunder-creative/artifacts/runs/`:

| ID | Endpoint | Observed bytes |
|---|---|---:|
| `ffull_s3_ep1` | `ffull_s3/model_ep1.pt` | 1,053,479 |
| `ffull_s4_ep1` | `ffull_s4/model_ep1.pt` | 1,053,479 |
| `frawW_s3_ep3` | `frawW_s3/model_ep3.pt` | 1,054,943 |
| `frawW_s4_ep3` | `frawW_s4/model_ep3.pt` | 1,054,943 |
| `frawW_s3ft_ep1` | `frawW_s3ft/model_ep1.pt` | 1,053,607 |
| `frawW_s4ft_ep1` | `frawW_s4ft/model_ep1.pt` | 1,053,607 |

Their combined observed size is 6,324,058 bytes. These are historical `is_file`/size observations, not content identities: no checkpoint SHA-256 was calculated. Before any later metadata load, root must freshly recheck each exact path, size, `lstat` identity, resolved target and full SHA-256. The probe's `is_file` checks do not establish that the logical paths are non-symlinks. The existing reviewed metadata helpers reject symlink inputs; if a fresh check finds symlinks, preserve both logical and resolved paths and use an explicitly reviewed link-aware allowlist rather than changing these helpers silently.

The two `ffull` logs name `fmu128_s3@ep3` and `fmu128_s4@ep3` as initialization parents. The W18 probe did not stat those parent endpoints, so this packet does not assert their file paths, existence or sizes and does not include them as inputs. They remain a separate exact-path/stat preflight dependency. Logs are configuration evidence, not a substitute for checkpoint identity.

## Existing helper compatibility

Neither reviewed extractor can consume these six endpoints unchanged. `tl3_checkpoint_metadata_r1/extract_checkpoint_metadata.py` enforces a hard-coded eight-file allowlist and requires exactly those eight IDs, paths and sizes. `tl3_fin256_metadata_r1/extract_fin256_metadata.py` enforces its distinct two-file allowlist and source-observation binding. Both reject inputs outside their exact contracts. Their bounded design is a useful reference, but changing either sealed helper would invalidate its reviewed contract and tests. This packet therefore contains a non-runnable root-binding input template, not a new extractor. Any later loader needs its own exact input allowlist and review, or a separately versioned and tested extension; it must use `torch.load(..., map_location="cpu", weights_only=True)` and stop on unsupported objects without an unrestricted fallback.

## Bounded future output contract

If a later owner chooses to prepare a loader, keep it CPU-only under the already used support-job envelope: 0.5 CPU core, at most 1.5 GB memory, zero swap and a 120-second hard limit. Hide CUDA before importing Torch and verify CUDA remains uninitialized. Root must freshly check campaign/worker state and resource headroom, select a physical disk-backed `TMPDIR` on B, and bind a fresh admission receipt; this packet does not perform those checks or authorize execution.

The input template caps each checkpoint at 32 MiB and the aggregate at 64 MiB, requires fresh per-file SHA-256 pins and leaves both parent checkpoints unresolved. A successful future run should hash each source before and after restricted loading, verify stable file generation, summarize only safe scalar/config fields and tensor key/shape/dtype metadata, discard each loaded object before opening the next, and serialize no tensor values. Capture the exact output JSON plus raw stdout and stderr bytes to new paths, recording their byte counts and SHA-256 values. Do not copy checkpoint payloads to the Mac or to the output receipt. Keep all output paths create-only and preserve any failed attempt evidence.

The six inputs in [INPUT_MANIFEST_TEMPLATE_R1.json](INPUT_MANIFEST_TEMPLATE_R1.json) have null hashes and no root admission binding, so the template is intentionally not runnable. No test or loader command is claimed in this preparation-only packet. Its source and helper bindings are sealed in [PREPARATION_MANIFEST_R1.json](PREPARATION_MANIFEST_R1.json); that manifest excludes its own hash.
