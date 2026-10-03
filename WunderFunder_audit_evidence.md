# WunderFunder research audit — evidence index

Audit date: 2 October 2026. Sources are the three archives uploaded in this conversation.

C = `WunderFunder-research-creative-iteration-20260930` source snapshot. R = the newer `research-review/research` tree. Q = `WunderFunder-quant-probes` (earlier overlapping source snapshot). The latest R ledger contains 323 entries; its reference recipe is not identical to C/V1.

## Scope
This is a source/artifact review, a selected data-free test run, and synthetic reproduction of specific implementation properties. The market dataset, prediction caches and trained checkpoint bodies were not available in the uploaded snapshots for independent retraining or score reproduction. Reported competition results are archive records, not independently fetched leaderboard results. No new market-data performance claim is made.

Selected existing tests executed: `test_gmetric.py`, `test_dual_clock.py`, `test_paired_seed_eval.py`, `test_conditional_teacher_probe.py`. Result: **39 passed**. This is not the entire test suite and does not validate every research path. New synthetic checks are in `reproduce_checks.py` and `synthetic_checks.json`.

## Finding identifiers
- E01: Current V1 result, chronology, metric and exposed-fold caveats.
- E02: Atom-head weighted-CE/readout mismatch; contrast the older correctly untilted readout.
- E03: Prefix-based head-only freeze leaves the event GRU trainable.
- E04: Chunk size changes optimizer/scheduler/update exposure, not only memory horizon.
- E05: Mixed global/block class diagnostics; convolution chunk-state boundary.
- E06: Newer candidate and distillation results; existing negative joint-head and pooled-memory tests.
- E07: Narrow architecture/feature negatives, atom results, old ceiling language, ensemble curve.
- E08: Matched-seed width result and under-covered conditional-teacher gate.
- E09: Later deep-audit qualifications on auxiliaries, clock decoding and sparse geometry.
- E10: Weak i1 leader rejection, row-thinning proposal and residual falsifiers.
- E11: Public-composition and forecast assumptions, invalid probability field.
- E12: Future-resolution-conditioned distillation proposal.
- E13: Completed Transformer control, matched mixture and confirmation limitations.
- E14: Independent synthetic/unit-test output.

Each source excerpt below has its original relative path, original line numbers, and SHA-256. Excerpts are not a substitute for the complete archives; they make the key review claims traceable.


## E01 — Best recorded package and latest branch caveats

Source: `C/research/CURRENT_STATE.md`

SHA-256: `e9415ebd87b6b14f5ab0e052b8d67c695f4f555b776577ef314e3b9999ea9229`


Lines 1–19:

```text
1: # Current research and complete experiment archive
2: 
3: Updated 1 October 2026 (Moscow). This is an index to the evidence; historical results remain in their original ledgers and journals.
4: 
5: ## Best recorded submission
6: 
7: **`candV1_ft2.zip`: public WP 0.6641, reported runtime 41m42s.** This is C's four-member ensemble with two epochs of head-only fine-tuning: distilled GRU256, GRU128 seed 3, GRU128 seed 4, and chunk-500 GRU128, equal weights, native C++ fp16 inference. The public result is recorded in the [September 29–30 journal](quant-probes-2026-09-29.md); it was not independently fetched from the leaderboard during this audit.
8: 
9: | Candidate | Reported public WP | Reported runtime |
10: |---|---:|---|
11: | A: two GRU256 members | 0.6629 | ~35m; governor dropped a member |
12: | B: GRU256 + two GRU128 | 0.6639 | ~43m |
13: | C: distilled GRU256 + two GRU128 + chunk-500 GRU128 | 0.6640 | 48m52s |
14: | V1: C with two-epoch head FT | **0.6641** | 41m42s |
15: | V2: validation repeated 3x in parent training | 0.6611 | 41m42s |
16: | V3: parents trained on train only, validation in head FT | 0.6629 | 49m26s |
17: 
18: The 0.0001 C→V1 difference is a recorded leaderboard difference, not established statistical superiority. Runtime varied substantially between submissions. V1 SHA256: `ea494921497d9f3cf53b19b1b68285133983560a249b38ef40738b3d846d319d`.
19: 
```


Lines 38–57:

```text
38: | Event attention, two-stream GRU, class heads, distributional/atom heads | No useful contribution to the tested deployed blend; see individual controls and folds in the journal. |
39: | Decoded price levels, discrete class-3 tables, residual corrections and raw additional-feature probes | Tested variants did not yield consistent cross-fold gains. This does not establish a general information-theoretic ceiling. |
40: | Level augmentation/drop, sequence V-REx, periodic inference resets, excessive validation upweighting | Negative in the tested protocols. Training-only random-origin augmentation would be a different, still untested intervention. |
41: | Matured-label feedback | An apparent large gain used labels before their actual maturity. The corrected quote-event-clock version was null; the invalid result remains documented. |
42: | Causal segment-recurrent Transformer | Both completed FT variants were tested at a fixed 20% teacher weight beside C: pooled gains +0.000115 / +0.000126, both negative on fold 1. Neither passed the registered +0.0005/both-fold-positive allocation gate. No new GPU run launched. |
43: | Runtime governor | Heterogeneous model costs are approximated as equal in current policy. A cost-model discrepancy alone does not prove a score/runtime improvement. |
44: 
45: ## Current creative-research iteration
46: 
47: The [registered iteration](iterations/2026-09-30-creative/) fixes three methodological problems: separate reused-fold discovery from confirmation and training-seed variation from chain uncertainty; optimize the user's competition objective instead of imposing novelty; replace categorical role vetoes with empirical comparisons and leakage checks.
48: 
49: M01 re-evaluated existing matched width-256/128 head-FT runs for seeds 3/4/5, fixed epochs, on 860 pure sequences in 250 chains. Candidate-minus-control deltas were **+0.000073, −0.000339, +0.000792**; mean **+0.000175**, sample SD **0.000572**. The three-seed mean-prediction ensemble improved by **+0.000696**, a different estimand. These results do not establish a stable positive per-model width effect. See the [immutable result and input hashes](iterations/2026-09-30-creative/M01-paired-seeds.result.json).
50: 
51: I1 tested one preregistered disagreement-gated teacher mixture against a fixed mixture of the same models. Its pooled improvement was only **+0.000090**, below the **+0.0004** allocation threshold; gated target-weight coverage was **0.81% / 0.80%**, below the required **5%** in each fold. The specific screen failed its gate. No threshold was retuned and no student training was launched. See the [result](iterations/2026-09-30-creative/I1-result.json) and [completed iteration report](iterations/2026-09-30-creative/report.md).
52: 
53: The [next quant round](iterations/2026-10-01-next/report.md) prioritizes a fixed-shape, main-only head fine-tune: current auxiliary losses remain active even when the GRU trunk is frozen. Two lower-priority feasibility studies examine actual decoded quote-event clocks and sparse volume-update coordinates before new architecture/feature training. The round also records cache cleanup, restoration mappings, and the two completed Transformer comparisons. These proposals have no new training result yet.
54: 
55: The [deeper three-branch audit](iterations/2026-10-01-deep/report.md) now supplies three completed training-side CPU diagnostics and detailed controlled study designs. Shared main/combined-aux gradients were positively aligned in two early windows (+.068/+.294), while global clipping coupling was measured; the generic conflict explanation is unsupported at these points. A new train-only class-3 decoder removes 35.67% of proxy event writes, but signed-key agreement/support gates prevent treating it as a certified quote clock; classes 0–2 remain entirely on fallback. The fixed sparse-volume dictionary failed its geometry gate (6.91–11.93% low-error coverage versus 80% required), despite passing simple planted sparse-data controls. No new GPU fit or validation WP query ran. Next training should use two-epoch V1-like anchors, not silently replace them with one-epoch C; ensemble-aware head FT is a separate reserve. The architecture alternative is row-GRU state rather than instantaneous projection as event-GRU input, with explicit freezing of `event_gru` during head-only FT. See the three dossiers and independent cross-reviews for limitations and costs.
56: 
57: All validation folds have prior exposure. A pure chain excludes a training neighbour; it is not a fresh holdout. Chain bootstrap intervals condition on the fitted models, do not measure optimizer variation, and do not correct adaptive selection or hidden-test shift. Read recent pooled-WP comparisons separately from older block-averaged reports; historical scorers and definitions are retained with the source snapshots. On 1 October the official data overview and FAQ were retrieved through web search and explicitly confirmed global pooled scoring. This resolves the previous primary-source limitation: see [current metric](../METRIC.md) and [verification record](iterations/2026-10-01-next/metric-source.json). Historical reports retain what was known at their execution time.
```


## E01 — Current metric specification

Source: `C/METRIC.md`

SHA-256: `2b264898016818a2f81aa441ac8e0f681e39254b0fd7926a9878ded601ad450d`


Lines 1–20:

```text
1: # Current scoring reference: global weighted Pearson
2: 
3: Verified against the organizer's [data overview](https://wundernn.io/connectome/docs/data_overview) and [FAQ](https://wundernn.io/connectome/docs/faq) on 1 October 2026. The current score pools selected rows across sequences. It does not average per-sequence correlations or apply the old block-eligibility filter.
4: 
5: For each target separately:
6: 
7: 1. Select `need_prediction AND is_scored` rows from the evaluated dataset.
8: 2. Clip targets and predictions to `[-2, 2]`.
9: 3. Use the absolute clipped target as the row weight.
10: 4. Compute weighted Pearson correlation over all selected rows together.
11: 
12: The final score is the arithmetic mean of the correlations for `t0` and `t1`. Higher is better. Average ensemble predictions before the scoring clip. A pooled score cannot generally be recovered by averaging fold or sequence correlations.
13: 
14: The current research implementation is [src/wnn/gmetric.py](src/wnn/gmetric.py): `global_wp` for arrays and `GlobalAccumulator` for streaming. [scripts/fold_eval.py](scripts/fold_eval.py) evaluates selected validation chains using the same six sufficient statistics per target. Computation uses float64 accumulation; local implementations return zero for degenerate sides.
15: 
16: Model state must still reset at each new sequence and update on every row, including the 99 warm-up rows. The hidden scoring mask is unavailable at inference. These stream rules are independent of the global score aggregation.
17: 
18: The imported [block metric document](METRIC_BLOCK_HISTORICAL.md), `utils.BlockAccumulator`, and `src/wnn/metric.py` are retained for reproducing historical experiments. They implement a different aggregation and must not be used to label a new score as the current competition metric. Their tests and old results have not been rewritten.
19: 
20: See the [source-verification record](research/iterations/2026-10-01-next/metric-source.json). Documentation verifies the organizer's declared metric; it is not an audit of the platform's internal scorer.
```


## E02 — Atom-head decoder

Source: `C/src/wnn/atom_head.py`

SHA-256: `6ad8cef658f81a4895557c25a4ba88f8fb1ce513a6e6c43cc5001101adcc91a5`


Lines 1–44:

```text
1: """RNNModel with a class-3 distributional ("atom") head and a metric-optimal decode (research, --atom-head).
2: 
3: Class-3 targets take few joint values (t0, t1) (64 atoms cover 99.9% of the metric weight, cache/class3_atoms.npy).
4: The atom head predicts a categorical distribution pi over the atoms (trained with |y|-weighted cross-entropy on
5: class-3 rows). Its decode is the per-target optimum of the |y|-weighted squared error:
6:     d_j = sum_k pi_k g_kj |g_kj| / sum_k pi_k |g_kj|
7: The output mixes the regression head and the decode: y_hat = reg + lambda * p3 * (d - reg), with p3 the model's own
8: class-3 probability (aux class logits) and lambda = sigmoid(learned scalar). The atom logits are appended at the
9: end of the output vector for the loss; prediction uses the first two columns only.
10: """
11: 
12: from __future__ import annotations
13: 
14: import torch
15: from torch import nn
16: 
17: from wnn.nn import RNNModel
18: 
19: 
20: class AtomRNNModel(RNNModel):
21:     def __init__(self, n_in, hidden, layers, dropout, n_out, class_slice, atoms, **kw):
22:         super().__init__(n_in, hidden, layers, dropout, n_out=n_out, **kw)
23:         self.class_slice = class_slice
24:         self.register_buffer("atoms", torch.as_tensor(atoms, dtype=torch.float32))  # (K, 2)
25:         k = self.atoms.shape[0]
26:         self.atom_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, k))
27:         self.mix_logit = nn.Parameter(torch.zeros(()))
28: 
29:     def forward(self, x, h=None):
30:         z = self.encode((x - self.mean) / self.std)
31:         o, h = self.gru(z, h)
32:         feats = torch.cat([o, z], dim=-1)
33:         out = self.head(feats).float()
34:         logits = self.atom_head(feats).float()
35:         pi = torch.softmax(logits, dim=-1)
36:         g = self.atoms  # (K, 2)
37:         ga = g.abs()
38:         dec = (pi @ (g * ga)) / (pi @ ga).clamp_min(1e-6)  # (B, T, 2)
39:         a, b = self.class_slice
40:         p3 = torch.softmax(out[..., a:b], dim=-1)[..., 3:4]
41:         lam = torch.sigmoid(self.mix_logit)
42:         reg = out[..., :2]
43:         mixed = reg + lam * p3 * (dec - reg)
44:         return torch.cat([mixed, out[..., 2:], logits], dim=-1), h
```


## E02 — Atom classification weighting

Source: `C/scripts/train_rnn.py`

SHA-256: `f97047e6ef51f32097f2cbe2eefc7c57766771ebaed12ae8798a3c1b7efcd52f`


Lines 982–994:

```text
982:                 if args.atom_head > 0:
983:                     atoms_t = model.atoms if hasattr(model, "atoms") else None
984:                     if atoms_t is not None:
985:                         yv = y[:, sl].float()
986:                         dist = (yv[..., None, :] - atoms_t[None, None]).abs().sum(-1)  # (B, T, K)
987:                         dmin, aidx = dist.min(-1)
988:                         c3 = (sequence_class(y) == 3)[:, None].expand_as(dmin)
989:                         wce = main_valid[:, sl].float() * yv.clamp(-2, 2).abs().mean(-1) * (c3 & (dmin < 1e-3)).float()
990:                         logit = out[..., -atoms_t.shape[0]:].float()
991:                         ce = F.cross_entropy(logit.reshape(-1, logit.shape[-1]), aidx.reshape(-1), reduction="none")
992:                         loss = loss + args.atom_head * (ce * wce.reshape(-1)).sum() / wce.sum().clamp_min(1e-8)
993:                 if args.vrex > 0 and epoch >= args.vrex_start:
994:                     # V-REx (2003.00688): penalise the variance of per-sequence normalised risks, so the model relies
```


## E02 — Older outcome readout explicitly corrects CE tilting

Source: `C/src/wnn/outcomes.py`

SHA-256: `47625c6712975f3611b92186eae9adf5ccc0b2b64e70bf9e3387935969a9be75`


Lines 90–110:

```text
90:         return (self.m1 / n).float(), (self.m0 / n).float()
91: 
92: 
93: def readout(logits: torch.Tensor, m1: torch.Tensor, m0: torch.Tensor, tilted: bool = False) -> torch.Tensor:
94:     """(..., N_CODES) logits -> (..., 2) tilted-mean predictions.
95: 
96:     With ``tilted`` the head was trained with |y|-weighted cross-entropy, so its softmax is
97:     pi_j * mbar_j (mbar = mean |y| of the code over both targets); dividing by mbar recovers pi.
98:     """
99:     pi = torch.softmax(logits.float(), dim=-1)
100:     if tilted:
101:         pi = pi / (0.5 * (m0[:, 0] + m0[:, 1])).clamp_min(1e-3)
102:     return (pi @ m1) / (pi @ m0).clamp_min(1e-6)
103: 
104: 
105: N_SP_AUX = 12  # auxiliary spread head: 1..11 ticks, 12 = 12 or more
106: 
107: 
108: def structure_labels(y: torch.Tensor, cls: torch.Tensor):
109:     """(B, T, 2) targets, (B,) classes -> spread bin (0..11), branch (0 below, 1 inside, 2 above), class per row."""
110:     q = _quanta(y.device, y.dtype)[cls][:, None, None]
```


## E03 — Head-only parameter selection

Source: `C/scripts/train_rnn.py`

SHA-256: `f97047e6ef51f32097f2cbe2eefc7c57766771ebaed12ae8798a3c1b7efcd52f`


Lines 794–809:

```text
794:         init = torch.load(ROOT / "artifacts" / "runs" / init_run / init_file, map_location="cpu", weights_only=False)
795:         model.load_state_dict(init["model"])
796:     else:
797:         fit_normalization(model, train_loader, device)
798:     frozen = tuple(v for v in args.freeze.split(",") if v)
799:     for pname, prm in model.named_parameters():
800:         if frozen and pname.startswith(frozen):
801:             prm.requires_grad_(False)
802:     opt = torch.optim.AdamW([prm for prm in model.parameters() if prm.requires_grad], lr=args.lr, weight_decay=args.wd,
803:                             fused=bool(args.fused_adam))
804:     total = len(train_loader) * math.ceil(20_000 / args.chunk) * args.epochs
805:     sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
806:                                                 pct_start=0.05, final_div_factor=100)
807:     amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
808:     scaler = torch.amp.GradScaler(enabled=args.precision == "fp16")
809:     swa_model = torch.optim.swa_utils.AveragedModel(model) if args.swa_frac > 0 else None
```


## E03 — Default fold fine-tuning flags

Source: `C/scripts/fold_ft.sh`

SHA-256: `f4392dd8091c1e6e51ee78102103eb3d2bac7885a71d7b8c4c0438030203cab2`


Lines 1–26:

```text
1: #!/usr/bin/env bash
2: # Two-stage gcorr LP-FT of a fold parent on validation rows with true masks: the fine-tune stage of the
3: # public-champion recipe (scripts/ft_gclp.sh), with fold_eval.py naming.
4: #   scripts/fold_ft.sh NAME PARENT FOLD GPU SEED [extra flags matching the parent, e.g. --hidden 256]
5: # Parent: artifacts/runs/PARENT_fFOLD/model_ep${PARENT_EP:-3}.pt (fixed final epoch). Trains on validation chains outside
6: # the held-out fold (head first with the trunk frozen, then everything), predicts all validation sequences
7: # into cache/pred_NAME_fFOLD_valid.npy. Log: artifacts/logs/NAME_fFOLD.log. FT2_LR sets the full-network
8: # stage learning rate (default 2e-4, the champion value).
9: set -euo pipefail
10: cd "$(dirname "$0")/.."
11: name=$1 parent=$2 fold=$3 gpu=$4 seed=$5; shift 5
12: out="${name}_f${fold}"
13: log="artifacts/logs/${out}.log"
14: export CUDA_VISIBLE_DEVICES=$gpu WNN_GPU_FRACTION=${WNN_GPU_FRACTION:-0.5} WNN_LOADER_RAM_GB=4 WNN_LOADER_WORKERS=2
15: export WNN_CPU_PCT=${WNN_CPU_PCT:-8} WNN_MEM_MAX=${WNN_MEM_MAX:-16G}
16: common=(--train-source valid --n-valid-folds 4 --full-valid --loss gcorr --mask-weight --batch 16 --epochs 1
17:         --aux-struct 0.3 --aux-next 0.3 --hidden 128 --seed "$seed" --fast
18:         --valid-fold "$fold" --eval-source vfold "$@")
19: echo "START $(date -Is) gpu=$gpu $out parent=${parent}_f${fold}@ep${PARENT_EP:-3} ${*}" >> "$log"
20: scripts/limited .venv/bin/python scripts/train_rnn.py --name "${out}h" --init "${parent}_f${fold}@ep${PARENT_EP:-3}" \
21:   --freeze "${FT_FREEZE:-inp,gru}" --lr 5e-4 "${common[@]}" >> "$log" 2>&1
22: scripts/limited .venv/bin/python scripts/train_rnn.py --name "$out" --init "${out}h" --lr "${FT2_LR:-2e-4}" "${common[@]}" >> "$log" 2>&1
23: scripts/limited .venv/bin/python scripts/predict_run.py "$out" valid >> "$log" 2>&1
24: # head-only stage (trunk frozen) as its own candidate: cache/pred_NAMEH_fFOLD_valid.npy
25: scripts/limited .venv/bin/python scripts/predict_run.py "${out}h" valid --output-name "${name}H_f${fold}" >> "$log" 2>&1
26: echo "DONE $(date -Is) $out" >> "$log"
```


## E03 — Event branch parameter names

Source: `C/src/wnn/nn.py`

SHA-256: `160447df28fa090684757551a75ee61d4f35c8c8c16c246324565306f46b8ec4`


Lines 191–253:

```text
191: class DualClockModel(nn.Module):
192:     """Row-clock GRU plus an event-clock GRU that steps only on i0 price events.
193: 
194:     ``forward(x, events, state)`` with x (B, T, n_in), events (B, T) bool.
195:     State ``(h_row, h_event, gap)`` carries across TBPTT chunks; ``gap`` counts
196:     rows since the last event. The event GRU input at an event row is the
197:     projected row input plus log1p(rows since the previous event). Rows between
198:     events see the event state from the latest event at or before them.
199:     """
200: 
201:     def __init__(self, n_in: int, hidden: int = 256, layers: int = 2, event_hidden: int = 128,
202:                  dropout: float = 0.0, n_out: int = 2):
203:         super().__init__()
204:         self.register_buffer("mean", torch.zeros(n_in))
205:         self.register_buffer("std", torch.ones(n_in))
206:         self.inp = nn.Linear(n_in, hidden)
207:         self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
208:                           dropout=dropout if layers > 1 else 0.0)
209:         self.event_gru = nn.GRU(hidden + 1, event_hidden, num_layers=1, batch_first=True)
210:         self.event_hidden = event_hidden
211:         self.head = nn.Sequential(nn.Linear(2 * hidden + event_hidden, hidden), nn.GELU(),
212:                                   nn.Linear(hidden, n_out))
213: 
214:     def forward(self, x, events, state=None):
215:         b, t, _ = x.shape
216:         dev = x.device
217:         z = F.gelu(self.inp((x - self.mean) / self.std))
218:         if state is None:
219:             h_row = None
220:             h_event = torch.zeros(b, self.event_hidden, device=dev, dtype=z.dtype)
221:             gap_in = torch.zeros(b, device=dev, dtype=torch.long)
222:         else:
223:             h_row, h_event, gap_in = state
224:         o, h_row = self.gru(z, h_row)
225: 
226:         rows = torch.arange(t, device=dev).expand(b, t)
227:         neg = torch.full_like(rows, -1)
228:         last = torch.cummax(torch.where(events, rows, neg), dim=1).values
229:         prev = torch.cat([neg[:, :1], last[:, :-1]], dim=1)
230:         gap = torch.where(prev >= 0, rows - prev, rows + 1 + gap_in[:, None])
231:         counts = events.sum(1)
232:         n_upto = torch.cumsum(events.long(), 1)
233: 
234:         held = h_event[:, None, :].expand(b, t, self.event_hidden)
235:         if int(counts.max()) > 0:
236:             maxc = int(counts.max())
237:             order = torch.argsort((~events).to(torch.int8), dim=1, stable=True)[:, :maxc]
238:             ev_in = torch.cat([
239:                 torch.gather(z, 1, order[..., None].expand(b, maxc, z.shape[-1])),
240:                 torch.log1p(torch.gather(gap, 1, order).to(z.dtype))[..., None],
241:             ], dim=-1)
242:             ev_out, _ = self.event_gru(ev_in.to(h_event.dtype), h_event[None].contiguous())
243:             ev_out = ev_out.to(z.dtype)
244:             idx = (n_upto - 1).clamp_min(0)
245:             gathered = torch.gather(ev_out, 1, idx[..., None].expand(b, t, self.event_hidden))
246:             held = torch.where((n_upto > 0)[..., None], gathered, held)
247:             last_idx = (counts - 1).clamp_min(0)
248:             h_last = ev_out[torch.arange(b, device=dev), last_idx]
249:             h_event = torch.where((counts > 0)[:, None], h_last, h_event)
250: 
251:         last_row = last[:, -1]
252:         gap_out = torch.where(last_row >= 0, t - 1 - last_row, gap_in + t)
253:         out = self.head(torch.cat([o, held, z], dim=-1))
```


## E04 — Chunk update/scheduler construction and detach

Source: `C/scripts/train_rnn.py`

SHA-256: `f97047e6ef51f32097f2cbe2eefc7c57766771ebaed12ae8798a3c1b7efcd52f`


Lines 800–810:

```text
800:         if frozen and pname.startswith(frozen):
801:             prm.requires_grad_(False)
802:     opt = torch.optim.AdamW([prm for prm in model.parameters() if prm.requires_grad], lr=args.lr, weight_decay=args.wd,
803:                             fused=bool(args.fused_adam))
804:     total = len(train_loader) * math.ceil(20_000 / args.chunk) * args.epochs
805:     sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
806:                                                 pct_start=0.05, final_div_factor=100)
807:     amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
808:     scaler = torch.amp.GradScaler(enabled=args.precision == "fp16")
809:     swa_model = torch.optim.swa_utils.AveragedModel(model) if args.swa_frac > 0 else None
810:     ema_model = None
```


Lines 938–970:

```text
938:                 td_a = args.td_alpha * min(1.0, step_count / ramp_steps)
939:             timer.mark("labels")
940:             h = h_ema = None
941:             for s in range(0, inputs.shape[1], args.chunk):
942:                 sl = slice(s, s + args.chunk)
943:                 if args.state_reset and s % args.state_reset == 0:
944:                     h = h_ema = None
945:                 with torch.autocast("cuda", dtype=amp_dtype):
946:                     out, h = forward_chunk(model, inputs, sl, h)
947:                 h = detach_state(h)
948:                 timer.mark("forward")
949:                 if args.ema_teacher > 0 or args.td_alpha > 0:
950:                     with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype):
951:                         out_ema, h_ema = forward_chunk(ema_model.module, inputs, sl, h_ema)
952:                     h_ema = detach_state(h_ema)
953:                 if moments is not None:
954:                     ce = F.cross_entropy(out.float().reshape(-1, N_CODES), codes[:, sl].reshape(-1), reduction="none")
955:                     wv = ce_w[:, sl].reshape(-1)
956:                     loss = (ce * wv).sum() / wv.sum().clamp_min(1e-8)
957:                     if args.ce_mse > 0:
958:                         pred = readout(out, *moments.tables(), tilted=args.ce_tilt)
959:                         loss = loss + args.ce_mse * weighted_mse_loss(pred, y[:, sl], main_valid[:, sl])
960:                 elif teach is not None:
961:                     yc = y[:, sl].float().clamp(-2, 2)
962:                     tgt = (1 - args.distill_alpha) * yc + args.distill_alpha * teach[:, sl].float().clamp(-2, 2)
963:                     w = yc.abs() * main_valid[:, sl][..., None]
964:                     loss = (w * (out[..., :2].float() - tgt) ** 2).sum() / w.sum().clamp_min(1e-8)
965:                 elif args.td_alpha > 0:
966:                     yc = y[:, sl].float().clamp(-2, 2)
967:                     s_end = min(s + args.chunk, inputs.shape[1])
968:                     jg = torch.clamp(nxt_all[:, sl], max=s_end) - 1  # last row of the constant-label run inside the chunk
969:                     rows = torch.arange(s, s_end, device=device)[None]
970:                     teach_t = torch.gather(out_ema[..., :2].float(), 1, (jg - s).clamp(min=0)[..., None].expand(-1, -1, 2))
```


Lines 1060–1094:

```text
1060:                         * y[:, sl].float().clamp(-2, 2).abs().mean(-1, keepdim=True)
1061:                     )
1062:                     quote_error = (out[..., quote_at:quote_at + 2].float() - torch.nan_to_num(label)) ** 2
1063:                     loss = loss + args.aux_quote * (quote_error * quote_weight).sum() / quote_weight.sum().clamp_min(1e-8)
1064:                 for j, (ya, va) in enumerate(aux):
1065:                     loss = loss + args.aux_weight * objective(out[..., 2 + 2 * j : 4 + 2 * j], ya[:, sl], va[:, sl])
1066:                 if args.ema_teacher > 0:
1067:                     cw = main_valid[:, sl].float()[..., None]
1068:                     consistency_scale = 1.0
1069:                     if args.ema_metric_weight:
1070:                         cw = cw * y[:, sl].float().clamp(-2, 2).abs()
1071:                         consistency_scale = 2.0  # preserve the original two-target loss scale
1072:                     loss = loss + args.ema_teacher * consistency_scale * (cw * (out[..., :2].float() - out_ema[..., :2].float()) ** 2).sum() / cw.sum().clamp_min(1e-8)
1073:                 timer.mark("loss")
1074:                 opt.zero_grad(set_to_none=True)
1075:                 scaler.scale(loss).backward()
1076:                 timer.mark("backward")
1077:                 scaler.unscale_(opt)
1078:                 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
1079:                 scaler.step(opt)
1080:                 scaler.update()
1081:                 sched.step()
1082:                 timer.mark("optimizer")
1083:                 if args.parity_dump:
1084:                     losses_trace.append(loss.detach().float())
1085: 
1086:                 step_count += 1
1087:                 if swa_model is not None and step_count >= swa_start:
1088:                     swa_model.update_parameters(model)
1089:                 if ema_model is not None:
1090:                     ema_model.update_parameters(model)
1091:                 if duty is not None:  # explicit rate limit (WNN_GPU_DUTY < 1): sync, then sleep
1092:                     torch.cuda.synchronize()
1093:                     duty.tick()
1094:                 elif not args.sync_free:
```


## E05 — Headline global score and block-averaged class diagnostics

Source: `C/scripts/train_rnn.py`

SHA-256: `f97047e6ef51f32097f2cbe2eefc7c57766771ebaed12ae8798a3c1b7efcd52f`


Lines 409–454:

```text
409:     if args_dict.get("class_heads"):
410:         # class logits sit after the targets, shifted-target aux outputs, spread (N_SP_AUX) and branch (3) logits
411:         start = 2 + 2 * len([v for v in str(args_dict.get("aux", "")).split(",") if v]) + N_SP_AUX + 3
412:         class_heads = (start, start + 4)
413:     return RNNModel(n_in, args_dict["hidden"], args_dict["layers"], args_dict["dropout"], n_out=n_out,
414:                     cell=args_dict.get("cell", "gru"), enc=args_dict.get("enc", "linear"),
415:                     tok_dim=args_dict.get("tok_dim", 32), att_layers=args_dict.get("att_layers", 1),
416:                     conv_k=args_dict.get("conv_k", 16), class_heads=class_heads)
417: 
418: 
419: @torch.no_grad()
420: def predict(model, x, chunk, device):
421:     model.eval()
422:     inputs = build_inputs(x.to(device, non_blocking=True))
423:     state, outs = None, []
424:     if STATE_RESET:
425:         chunk = min(chunk, STATE_RESET)
426:     for s in range(0, inputs.shape[1], chunk):
427:         if STATE_RESET and s % STATE_RESET == 0:
428:             state = None
429:         with torch.autocast("cuda", dtype=torch.float16):
430:             out, state = forward_chunk(model, inputs, slice(s, s + chunk), state)
431:         outs.append(readout(out, *READOUT, tilted=READOUT_TILTED) if READOUT is not None else out[..., :2].float())
432:     return torch.cat(outs, dim=1).cpu().numpy()
433: 
434: 
435: def evaluate(model, loader, chunk, device):
436:     preds, ys, masks, classes = [], [], [], []
437:     for x, y, m in loader:
438:         preds.append(predict(model, x, chunk, device))
439:         ys.append(y.numpy())
440:         masks.append(m.numpy())
441:         classes += [market_class(t) for t in y.numpy()]
442:     p, y, m = np.concatenate(preds), np.concatenate(ys), np.concatenate(masks)
443:     res = score(y, p, m)
444:     res["per_seq_wp"] = res["weighted_pearson"]
445:     g = global_wp(y, p, m)
446:     res.update(global_t0=g["t0"], global_t1=g["t1"], weighted_pearson=g["wp"])
447:     res["frac_abs_pred_gt2"] = float((np.abs(p[m]) > 2).mean())
448:     per = block_scores(y, p, m).mean(axis=1)
449:     cls = np.array(classes)
450:     res["by_class"] = {int(c): round(float(np.nanmean(per[cls == c])), 4) for c in np.unique(cls)}
451:     return res
452: 
453: 
454: def main():
```


## E05 — Convolution re-pads at each encode call

Source: `C/src/wnn/nn.py`

SHA-256: `160447df28fa090684757551a75ee61d4f35c8c8c16c246324565306f46b8ec4`


Lines 66–93:

```text
66:     def encode(self, xn):
67:         u = self.inp(xn)
68:         if self.enc == "tokens":
69:             b, t, _ = xn.shape
70:             flat = xn.reshape(b * t, -1)
71:             toks = torch.stack([lin(flat[:, a:c]) for lin, (a, c) in zip(self.tok_in, self.groups)], dim=1)
72:             toks = self.att(torch.cat([self.cls.expand(b * t, -1, -1), toks], dim=1))
73:             pooled = torch.cat([toks[:, 0], toks[:, 1:].mean(1)], dim=-1)
74:             u = u + self.tok_out(pooled).reshape(b, t, -1)
75:         z = F.gelu(u)
76:         if self.enc == "mlp":
77:             a, g = self.enc_up(self.enc_norm(z)).chunk(2, dim=-1)
78:             z = z + self.enc_down(F.gelu(a) * torch.sigmoid(g))
79:         elif self.enc == "conv":
80:             c = F.pad(z.transpose(1, 2), (self.conv_k - 1, 0))  # causal; zero left context at chunk starts
81:             z = z + F.gelu(self.conv_pw(self.conv_dw(c).transpose(1, 2)))
82:         return z
83: 
84:     def forward(self, x, h=None):
85:         z = self.encode((x - self.mean) / self.std)
86:         o, h = self.gru(z, h)
87:         feats = torch.cat([o, z], dim=-1)
88:         out = self.head(feats)
89:         if self.class_heads is not None:
90:             a, b = self.class_heads
91:             prob = torch.softmax(out[..., a:b].float(), dim=-1)
92:             res = self.class_res(feats).float().reshape(*out.shape[:-1], b - a, 2)
93:             out = torch.cat([out[..., :2].float() + (prob[..., None] * res).sum(-2), out[..., 2:].float()], dim=-1)
```


## E05 — Real-data boundary effect already measured: do not exaggerate toy mismatch

Source: `R/experiments.jsonl`, selected IDs. SHA-256 `b394f6a980f73a941e6b98805670f05e893dbbdb6c71eecbb34acda4aa1f1cc0`.


Original line 175, `E-175`:

```json
{
  "id": "E-175",
  "date": "2026-09-27",
  "kind": "native engine: conv encoder port (worktree branch worktree-agent-adf2cdf4134c35cb5, commit aa34ce2, native/CONV_PORT.md)",
  "result": {
    "parity_max_abs": 1.7e-06,
    "latency_overhead": [
      0.045,
      0.07
    ],
    "chunk_reset_vs_streaming_wp": {
      "streaming": 0.67423,
      "reset4000": 0.67422,
      "reset2000": 0.67421
    }
  },
  "claim": "a conv member can be packaged (true causal streaming, parity 1.7e-6, +5-7% latency); merge only if conv passes on EMA-4 folds",
  "status": "ready (unmerged)"
}
```


## E06 — Newer controlled and candidate results

Source: `R/experiments.jsonl`, selected IDs. SHA-256 `b394f6a980f73a941e6b98805670f05e893dbbdb6c71eecbb34acda4aa1f1cc0`.


Original line 291, `E-291`:

```json
{
  "id": "E-291",
  "date": "2026-10-01",
  "kind": "N2 pooled-memory GCLP FT (decay .995) + head-only vs two-stage FT + shared-trunk parent/head-FT output mix (folds 0+1; holdout with final ab67 members)",
  "command": "campaign_job c20261001_ftpool_f01; screen_score pool_shared_f01.json, holdout_shared.json",
  "result": {
    "pool_two_stage_vs_ft_pair": [
      -0.001521,
      0.000674
    ],
    "pool_head_vs_head_pair": [
      -0.000414,
      0.000433
    ],
    "head_pair_vs_two_stage_pair": [
      0.000635,
      0.000431
    ],
    "shared_parent_head_vs_ft_pair": [
      0.000983,
      0.000623
    ],
    "shared_parent_head_vs_mix4": [
      1.2e-05,
      0.00022
    ],
    "extra_pool_heads": [
      -5.8e-05,
      0.00017
    ],
    "mild_mix_shared_vs_ft_pair": 0.001477,
    "holdout_shared_vs_incumbent": [
      0.000379,
      0.00055
    ],
    "holdout_head_pair_vs_incumbent": [
      0.000167,
      0.00041
    ],
    "holdout_mild_shared": 0.00087
  },
  "verdict": "REJECT N2 (H-055). Head-only FT >= two-stage. Shared-trunk parent+head-FT mix reproduces the 4-trunk mix gain at the incumbent's trunk cost; confirmation on folds 2/3 preregistered."
}
```


Original line 293, `E-293`:

```json
{
  "id": "E-293",
  "date": "2026-10-01",
  "kind": "N1 ensemble-aware joint head FT vs matched solo extra head FT (since pair, folds 0+1)",
  "command": "campaign_job c20261001_jpair_f01 (jobs/joint_pair_f01.sh); screen_score jpair_f01.json",
  "result": {
    "joint_minus_solo": [
      -9.9e-05,
      3.1e-05
    ],
    "joint_minus_reference": [
      0.000263,
      0.000472
    ],
    "solo_minus_reference": [
      0.000362,
      0.000468
    ],
    "joint_parent_minus_ref_parent": [
      0.000143,
      0.000244
    ],
    "pub143_joint_minus_solo": -0.000112,
    "seconds": 405
  },
  "verdict": "REJECT H-056: co-adapting heads on the average is slightly worse than independent heads (both folds negative); extra solo head epoch is within noise"
}
```


Original line 307, `E-307`:

```json
{
  "id": "E-307",
  "date": "2026-10-01",
  "kind": "PREREGISTERED folds 2+3 confirmation of parent+head output mixes and class-routed RSP2",
  "command": "screen_score parentmix_confirm_f23.json (prereg parentmix_confirm_f23_PREREG.json incl. amendments 1-2 made before f2/f3 files existed)",
  "result": {
    "shared_parent_head_vs_ft_pair": [
      -7.5e-05,
      0.000407
    ],
    "fold2": 0.000224,
    "fold3": -0.000574,
    "mix4_vs_ft_pair": [
      0.000225,
      0.000269
    ],
    "RSP2_vs_ft_pair": [
      0.000306,
      0.000332
    ],
    "RSP2_fold2": 0.000165,
    "RSP2_fold3": 0.000378,
    "RSP2_pub143": 0.00038,
    "class3_ft_pair": 0.44857,
    "class3_RSP2": 0.44993
  },
  "verdict": "SP2 NOT CONFIRMED (fails pooled gate, fold 3 negative). mix4 fails. RSP2 positive on both folds and class-floor safe but below the +0.0005 gate: directional support only."
}
```


Original line 311, `E-311`:

```json
{
  "id": "E-311",
  "date": "2026-10-01",
  "kind": "distilled GRU256 since student, fold 1 (scored after cutoff from the finished 05:35 job) + folds 0+1 under the routed framework",
  "command": "screen_score d256_f01.json",
  "result": {
    "d256_head_vs_s3_head": [
      0.001772,
      0.00098
    ],
    "fold1_d256_head_vs_s3_head": 0.003355,
    "SP2_D_vs_SP2": [
      0.000942,
      0.000307
    ],
    "fold1_SP2_D_vs_SP2": 0.001613,
    "RSP2_D_vs_RSP2": [
      0.000527,
      0.000222
    ],
    "fold1_RSP2_D": 0.000873,
    "pub143_RSP2_D_vs_RSP2": 0.00062,
    "fit_seconds": 2606
  },
  "verdict": "d256 trunk (merged parent+head-FT heads) in the class-3 group adds +0.00053 (CI excludes 0) on folds 0+1; post hoc over two arms; final d256 not trained (cancelled 04:53 pending this result) -> top next step: final d256 + package RSP2_D (expected runtime ~36-41 min i16 given ~46% class-3 sequences; worst case all class 3 ~57 min)"
}
```


Original line 315, `E-315`:

```json
{
  "id": "E-315",
  "date": "2026-10-01",
  "kind": "creative: winning-Kaggle mindset scout (5 linked writeups/notebooks + order-book competitions) and 5 OOF probes",
  "by": "cr_scout",
  "result": {
    "F1_cov_share_within_seq": ">0.90",
    "F1b_best_lambda": "0 in 16/16 class x fold",
    "F1c_best_gamma": "0 in 16/16",
    "F3_block_persistence_corr": [
      -0.07,
      0.08
    ],
    "F3b_matured_true_y_upper_bound_by_lag": {
      "8": "+0.0053..+0.0072",
      "12": "+0.0026..+0.0038",
      "24": "~0",
      "64": "~0"
    },
    "F4_cell_overlay": "-0.0003..-0.0008",
    "F5_tree_stacker": "-0.009..-0.015",
    "F5_member_mean": "+0.0007/+0.0010/+0.0004/-0.0001"
  },
  "verdict": "post-hoc sequence-level, cell overlays and stacking dead; matured-label feedback upper bound decays to 0 by lag 24 events (consistent with leakage, teammates' null) -> weak lead contingent on certifying exit times",
  "artifacts": [
    "research/campaign_2026-10-01/creative/cr_scout/MINDSET.md",
    "research/campaign_2026-10-01/creative/cr_scout/corpus"
  ]
}
```


Original line 316, `E-316`:

```json
{
  "id": "E-316",
  "date": "2026-10-01",
  "kind": "S4_D package: since s4 trunk (parent+gcorr head-FT merged) + final distilled GRU256 trunk (parent+gcorr head-FT merged), equal 0.5, int16, governor off",
  "command": "package_submission.py c20261001_S4D_i16 ...; packages/verify_all.sh; engine_fp16_check calib f32/i16",
  "result": {
    "sha256": "ab0f528755aa8ba2033fbe558a7e8e6000ce9fb77f6a1c12b6c245b0172fec00",
    "bytes": 6176165,
    "i16_vs_f32_max": 4.16e-05,
    "levels_bitwise": true,
    "image_py311": "PASS",
    "reset": "PASS",
    "kcyc_per_row": 380.0,
    "projected_public_min_cycles": 48.3,
    "local_evidence": {
      "4fold_vs_ft_pair": [
        0.001776,
        0.00045
      ],
      "4fold_S3_D": [
        0.001266,
        0.000509
      ],
      "holdout_vs_incumbent": [
        0.001645,
        0.000722
      ],
      "pub143_4fold": 0.002068
    },
    "forecast": {
      "mean_seed_chosen": 0.66323,
      "mean_seed_averaged": 0.66301,
      "P_gt_0.6641": -0.04999999999999999,
      "P_ge_0.666": "<1e-3"
    }
  },
  "verdict": "best verified package so far; expected public ~0.6630-0.6632; below 0.666 goal"
}
```


Original line 318, `E-318`:

```json
{
  "id": "E-318",
  "date": "2026-10-02",
  "kind": "upload variants: routed R4D (classes 0-2 -> FT s4 + d256 merged; class 3 -> s4 merged + d256 merged), D_only, FT4_D on 4 clean folds and holdout",
  "command": "screen_score upload_variants_f0123.json, upload_variants_holdout.json",
  "result": {
    "folds_vs_ft_pair": {
      "R4D": [
        0.002002,
        0.000422
      ],
      "D_only": [
        0.001721,
        0.000705
      ],
      "FT4_D": [
        0.00176,
        0.000311
      ],
      "R4D_vs_S4_D": [
        0.000227,
        7.7e-05
      ]
    },
    "holdout_vs_incumbent": {
      "R4D": [
        0.001991,
        0.000703
      ],
      "D_only": [
        0.0012,
        0.000922
      ],
      "FT4_D": [
        0.001621,
        0.000504
      ],
      "R4D_vs_S4_D": [
        0.000346,
        0.000138
      ]
    },
    "forecast": {
      "R4D": {
        "mean": 0.6634,
        "p80": [
          0.6623,
          0.6645
        ],
        "P_gt_0.6641": 0.215,
        "P_ge_0.6660": 0.0017758619615369712
      },
      "S4D": {
        "mean": 0.66323,
        "p80": [
          0.6621,
          0.6643
        ],
        "P_gt_0.6641": 0.154,
        "P_ge_0.6660": 0.0006097186309879648
      },
      "D_only": {
        "mean": 0.66294,
        "p80": [
          0.6618,
          0.664
        ],
        "P_gt_0.6641": 0.09,
        "P_ge_0.6660": 0.00019486211094735945
      },
      "RSP2": {
        "mean": 0.66274,
        "p80": [
          0.6617,
          0.6637
        ],
        "P_gt_0.6641": 0.041,
        "P_ge_0.6660": 1.5525647371030224e-05
      }
    }
  },
  "verdict": "R4D best runtime-feasible variant on folds and holdout (post hoc selection among variants; routing directional); forecast ~0.6634, still below 0.6641 in expectation"
}
```


Original line 320, `E-320`:

```json
{
  "id": "E-320",
  "date": "2026-10-02",
  "kind": "distillation strength alpha 0.7 vs 0.5 for the GRU256 since student (fold 0)",
  "command": "campaign_job c20261001_d256a07_f0; screen_score alpha07_f0.json",
  "result": {
    "D07_vs_D05_mix": [
      0.002331,
      0.001255
    ],
    "D07h_vs_D05h": [
      0.002812,
      0.001366
    ],
    "S4_D07_vs_S4_D05": [
      0.000868,
      0.000492
    ],
    "class3_D07_minus_D05": 0.0036,
    "pub143_D07_vs_D05": 0.002488
  },
  "verdict": "promising (fold 0 only): stronger distillation lifts the student materially; confirm on fold 1 and test alpha 0.85 before the final student"
}
```


Original line 323, `E-323`:

```json
{
  "id": "E-323",
  "date": "2026-10-02",
  "kind": "explanatory residual analysis of the R4D OOF analog (notebook F1-F7)",
  "result": {
    "loss_share_class3_runs_64_1024_rows": 0.499,
    "loss_share_rows_32_512_before_next_move": 0.436,
    "run_start_corr_long_runs_ge64": {
      "c0": 0.24,
      "c1": 0.21,
      "c2": 0.07,
      "c3": 0.02
    },
    "member_residual_corr_since_vs_d256": [
      0.985,
      0.999
    ],
    "alpha_mechanism": "alpha 0.7 vs 0.5 gain is concentrated in rows far (>=32 rows) from the next quote move, mainly class 3, where alpha 0.5 lost to the since member",
    "ft_vs_parent": "un-fine-tuned since parent is better early in long runs for classes 1-2"
  },
  "verdict": "actionable: (1) loss is in long class-3 runs where the outcome is unknown at run start; (2) members are near-identical in error (ensembling saturated); (3) distillation helps by denoising far-from-resolution labels -> try noise-adaptive distillation (H-059)",
  "artifacts": [
    "notebooks/eda_residuals_20261002.py",
    "research/campaign_2026-10-01/plots/eda/"
  ]
}
```


## E07 — Negative experiments and overbroad historical conclusions

Source: `C/research/quant-probes-2026-09-29.md`

SHA-256: `507b31da53f4f0890301d7bcea8c4945634511817d8258d5a0b57acc79768dde`


Lines 41–117:

```text
41: Reset every N rows in training and prediction (chunk 2000). Pure chains, folds 0+1, 860 sequences.
42: 
43: | variant | delta vs ref | boot SD | fold 0 | fold 1 | class 3 |
44: |---|---:|---:|---:|---:|---:|
45: | reset 2000 | -0.0007 | 0.0007 | -0.0002 | -0.0011 | -0.0019 |
46: | reset 4000 | -0.0009 | 0.0006 | -0.0007 | -0.0011 | -0.0031 |
47: 
48: Why it does not transfer: the reference WP is flat along the sequence (rows 99-1000: 0.699,
49: 5k-10k: 0.688, 15k-20k: 0.702), so there is no late-sequence state degradation to fix; the
50: competitor's model lacked the since-event counters and event features that carry long context here.
51: Artifacts: `research/artifacts/quant_probes/rst*_vs_ref.json`.
52: 
53: ### R2: order-flow sums over the quiet interval (`--quant-feats`, 31 inputs) — no gain
54: i0 volume codes minus their value at the last i0 price event (22), sums of trade-volume and
55: trade-price codes over trade rows since that event (8), log trade count. Batch/streaming parity exact.
56: 
57: | comparison | delta | boot SD | fold 0 | fold 1 |
58: |---|---:|---:|---:|---:|
59: | quant vs ref, all classes | -0.0002 | 0.0006 | -0.0002 | -0.0002 |
60: | within class 3 | +0.0008 | | | |
61: | classes 0/1/2 | -0.0007 / -0.0009 / -0.0002 | | | |
62: | class-3-routed (quant on class 3, ref elsewhere) vs ref | +0.0002 | 0.0005 | -0.0002 | +0.0005 |
63: 
64: Reading: the GRU already integrates this flow from per-row deltas; explicit window sums add
65: nothing measurable. The class-3 sign is the only hint, far inside noise. Routing tool:
66: `scripts/route_eval.py`.
67: 
68: ### Seed noise dominates single-seed deltas (methodology)
69: The same reference recipe with seed 4 vs seed 3: **+0.0009**, chain-bootstrap SD 0.0005,
70: P(delta <= 0) = 0.04 (fold 0 +0.0002, fold 1 +0.0014; class 3 +0.0020). A pure seed change passes
71: the old +0.0005 single-seed gate. The chain bootstrap measures sampling noise over sequences, not
72: optimisation noise, so single-seed deltas below ~0.001 are not evidence either way.
73: New protocol: compare **seed-pair averages** (the deployed ensemble form):
74: `fold_eval.py "cand+cand_s4" "ref+refs4"`. Reference pair 0.7000 vs singles 0.6989 / 0.6997.
75: 
76: Re-read with this in mind:
77: - reset: the two reset models averaged score 0.6983 vs reference pair 0.7000 (-0.0017; below both
78:   reference singles). Refuted with margin.
79: - quant i0: -0.0002 single seed, inconclusive; seed-4 run queued for a pair comparison.
80: 
81: ### R3: i1 cross-impact block (`--quant-i1`, +31 inputs) — negative (fold 0)
82: -0.0011 (SD 0.0010), class 3 -0.0040. Stopped before fold 1.
83: 
84: ### R4: class-3 expert (`--train-classes 3`, routed on class 3) — negative (fold 0)
85: -0.0016 (SD 0.0006, P(<=0) 0.997): the joint model beats a class-3 specialist trained on fewer
86: sequences. Stopped before fold 1.
87: 
88: ### R5: pooled weighted-Pearson + wMSE loss from scratch (`--loss wmse_gcorr`) — inside seed noise
89: +0.0003 vs reference seed 3 (SD 0.0005), -0.0006 vs reference seed 4. No evidence of an effect; not
90: replicated further.
91: 
92: ### R6: dual-clock GRU (`--arch dual`, row GRU + event-clock GRU) in the EMA regime — no gain
93: All fold sequences: fold 0 0.7001 vs 0.7005, fold 1 0.6752 vs 0.6765.
94: 
95: ### Engineering side result: C++ engine (branch `cpp-engine`, `native_cpp/`)
96: Parity 2.5e-6 vs PyTorch fp32 and 1.4e-6 vs the Rust engine. With fp16 dense weights on v3 it runs
97: hidden 128 at 17 us/row vs 27 (Rust) on this host, hidden 256 at 85 vs 156; fp16 changes WP by -1e-6 on
98: 60 sequences. A faster member means more members or a wider model inside the 60-minute budget.
99: 
100: ### Protocol fix: fixed final epoch
101: `train_rnn.py` saves `model.pt` as the best epoch **on the held-out fold** (selection on the evaluation
102: data). Only the hidden-256 runs peaked before the last epoch; their predictions were redone from
103: `model_ep3.pt`, and `fold_job.sh` now always predicts from the fixed final epoch (`EVAL_EPOCH`, default 3).
104: 
105: ### R7: capacity, hidden 256 (EMA-4 recipe) — small gain, not worth its cost
106: Fixed final epoch: +0.0006 vs the seed-3 reference (SD 0.0010; fold 0 +0.0007, fold 1 +0.0005;
107: classes 1/2 +0.0023/+0.0027, class 3 +0.0003). Epoch curve: +0.003-0.004 after epoch 0, shrinking to
108: +0.0005 by epoch 3, i.e. the EMA-4 hidden-128 model catches up. One hidden-256 member costs ~4x a
109: hidden-128 member in inference and scores -0.0006 below the two-seed hidden-128 pair (2x cost).
110: Seed-4 hidden-256 runs are queued to check a hidden-256 pair.
111: 
112: ### R2 follow-up: order-flow features, seed pair vs reference seed pair — refuted
113: quant(s3)+quant(s4) vs ref(s3)+ref(s4): **-0.0009** (SD 0.0006, P(<=0) 0.96), fold 0 -0.0007, fold 1
114: -0.0011, every class negative (class 3 -0.0009). Seed 4 alone: -0.0016. The single-seed class-3 hint was
115: noise; the explicit window sums slightly hurt (extra inputs, no new information).
116: 
117: ### Seed-ensemble curve (reference recipe, hidden 128, seeds 3-6) — saturates by K=3
```


Lines 281–339:

```text
281: # Round 3 (2026-09-30 night): bolder ideas
282: Three literature subagents (LOB microstructure; noisy labels / distillation / sequence models; competition
283: winners and quant practice) proposed 15 ideas; triage: mirror symmetry is impossible here (price codes are not
284: bid/ask symmetric: class-3 bid codes in [-2.27,-1.27], ask in [-0.84,-0.46]); online matured-label correction ~
285: Vlad's refuted E-080; distributional/CE heads ~ refuted E-071; full-network FT hurts h256 (above).
286: 
287: ## Privileged (non-causal) teacher distillation — no gain beyond ordinary distillation
288: Cross-fitted h128 teachers on train halves that also see raw rows t+L (`--lookahead`, `scripts/teacher_job.sh`,
289: `scripts/merge_teachers.py`). Teacher OOF valid WP: t+10,t+40 **0.813**; t+3,t+10 0.744 (causal ~0.70).
290: h128 students, fold 0, seed 3, head-FT, vs the h128 reference: t+10/40 alpha 0.5 +0.0007, alpha 0.8 +0.0011,
291: t+3/10 alpha 0.5 +0.0002; ordinary distillation from two causal h256 teachers +0.0010. A far more accurate
292: teacher does not make a better causal student: the distillation gain is smoothing, capped by what the causal
293: student can represent.
294: 
295: ## Blend weights (8 OOF members, cross-fitted between folds 0/1)
296: Global weights +0.00035 over equal weights on both folds (weights ~2/3 on the two d256 students); per-class
297: weights +0.0000. Per-class blending refuted.
298: 
299: ## Class-3 context table (Bayes-limit diagnostic, `scripts/c3_table.py`)
300: E_w[D | book-state code, log2 rows since last price event, last move sign, trade row] fitted on 1,500 class-3
301: train sequences: weighted corr with D on class-3 pure valid 0.157 / 0.127 (folds 0/1) vs the d256 pair 0.507 /
302: 0.476; adding the table to the GRU lowers corr at every beta. No discrete context structure is left unused.
303: 
304: ## Born-again generation 2 and longer distillation — plateau
305: Gen-2 h256 students, teacher = mean of four h256 models (two gen-1 students + two plain), alpha 0.7:
306: pair vs gen-1 pair +0.00036 (folds +0.0002/+0.0005, SD 0.0003); gen-2 in place of the gen-1 student in the C
307: blend -0.00006. Six epochs with distillation targets (fold 0): parent 0.7027 vs 0.7028 at four epochs (no
308: overfitting now, but no gain either).
309: 
310: ## Round-3 verdict
311: No breakthrough. Distillation is the only lever and it plateaus after one generation; privileged teachers,
312: per-class blending, discrete class-3 context tables and longer training add nothing measurable. The class-3
313: information ceiling for causal models of this family looks reached; remaining ensemble gains are ~+0.0003.
314: 
315: # Public: candC 0.6640 in 48m52s (candB 0.6639): the +0.0007 fold gain is invisible on the public LB.
316: 
317: ## New architecture: GRU + causal attention over the last 32 i0 price-event tokens (`--arch attn`)
318: `src/wnn/event_attn.py`: tokens (row projection + log gap) written at each i0 price event into a ring carried
319: across chunks; each row queries it from the GRU output (4 heads, d 64) and the head sees [GRU | proj | ctx].
320: Checked: chunked == full forward exactly, future perturbation leaves the prefix bit-identical.
321: h128, 4 epochs EMA: +0.0017/+0.0022/+0.0007 after epoch 0, final parents 0.7002 vs 0.7005 (f0), 0.6765 vs 0.6765
322: (f1). Head-FT: single -0.0002 (folds 0+1); fold-0 pair +0.0003 (SD 0.0004); as the 4th member in C instead of
323: c500 -0.0004. Refuted: faster convergence to the same ceiling, like width, distillation and privileged teachers.
324: 
325: ## Decoded class-3 price level in ticks (`scripts/eda_ticks.py`) — no residual information
326: Transition tables fitted on 1,500 class-3 train sequences (1,594 transitions; 18 unknown of 1.96M valid book
327: events). Features on the decoded mid path: expanding z, distance to rolling max/min (500/2000/8000 rows),
328: tick changes over 100/500/2000 rows, reversal fraction over the last 64 mid moves. Weighted corr with the
329: d256-pair residual on class-3 pure valid rows: |corr| <= 0.044 with signs flipping between folds; cross-fitted
330: ridge -0.012 / -0.006. No support/resistance, momentum or reversal-regime signal beyond the GRU.
331: 
332: # Status after round 3
333: Every direction tried since the 0.6640 submission converges to the same ceiling (~0.702 on folds 0+1 for the
334: best 4-member blend): width, distillation (incl. privileged and gen-2), attention over events, class heads,
335: per-class blending, discrete class-3 contexts, long-memory and decoded-level features.
336: 
337: ## Shift robustness (train-only protocol, `scripts/trainonly_job.sh`)
338: Train on train only, score all pure validation chains (streams never seen): reference 0.69566 (vs ~0.699 for
339: fold models that saw other validation chains: valid-like data is worth ~+0.003).
```


Lines 345–403:

```text
345: - `candV2_vr3.zip` (sha b13dfe87...9c75): all four members retrained with validation sequences repeated 3x.
346: Both ~C's runtime (~49 min).
347: 
348: ## LB 2026-09-30: V1 0.6641 (41m42s), **V2 0.6611 (-0.0029)**, both 41m42s (C 48m52s: scorer speed varies ~15%)
349: Upweighting validation 3x in parent training clearly hurts the test; a longer head fine-tune is neutral.
350: Next LB probe: `candV3_trainonly.zip` (sha f7943bc6...0fe): C's four members with parents trained on TRAIN ONLY
351: (`PARENT_SOURCE_MODE=train scripts/final_job.sh`), validation used only by the head fine-tune.
352: 
353: ## LB 2026-09-30: V3 (parents on train only) 0.6629 (49m26s). Valid weight in parents: none 0.6629, 1x (C) 0.6640,
354: 3x 0.6611 -> C is at the optimum of this knob.
355: 
356: ## Matured-label error feedback — the apparent +0.08 was LEAKAGE (caught)
357: `scripts/matured.py` (TRUE labels, "matured" 4 or 8 changes of the 22 price codes after the row): corr of the last
358: matured residual with the current residual 0.35-0.59, cross-fitted correction +0.084 / +0.094 (lag 4) and +0.019 /
359: +0.024 (lag 8). `scripts/matured_c3.py` with labels REBUILT from decoded ask/bid paths (known only at the 4th
360: ask/bid change): corr -0.04..+0.01, global effect -0.0003 / +0.0003. Code changes include book-shape changes that
361: do not move ask/bid, so "4 code events" came before the label was determined: the true-label residual carried
362: future information. Honest version: null (as Vlad's E-080). Lesson: maturity must be measured on the label's own
363: clock (decoded quote changes), never on a proxy event stream.
364: 
365: ## Round 4 (2026-09-30 afternoon)
366: - Two-stream GRU (separate i0 / i1 recurrent streams, `src/wnn/two_stream.py`, `--arch twostream`): pair vs h128 pair
367:   -0.0004 (folds +0.0011 / -0.0017, SD 0.0012); in C's blend +0.0001. Refuted.
368: - V-REx over sequences (`--vrex`, train-only -> unseen valid, seed pairs): beta 1 -0.0081 (SD 0.0006), beta 5 -0.114.
369:   (A first variant normalised by each chunk's target variance exploded.) Refuted.
370: - Data-detective subagent (CPU analyses on the ensemble residual, pure OOF, cross-fitted restacking): a0..a7 raw or
371:   as scalers, trade-side codes, i1 events since the last i0 event, per-sequence bias, class-3 rounding parity,
372:   cross-sequence state hashing (no shared streams) -> nothing survives on both folds. Diagnosis: pooled score depends
373:   strongly on the class-3 weight share (single-sequence chains are 64% class 3 and score 0.59 vs 0.74 for long
374:   chains); the local-public gap fits a class-3-heavier / shorter-stream test.
375: - Class-3 distributional atom head (`src/wnn/atom_head.py`, `--atom-head 0.5`; 64 joint (t0,t1) atoms cover 99.9% of
376:   class-3 metric weight; |y|-weighted CE + metric-optimal decode mixed with the regression head by p(class 3) and a
377:   learned lambda, which settled at 0.41): pair vs h128 pair -0.0008 (folds -0.0005/-0.0010, SD 0.0004, class 3
378:   -0.0009); in C's blend -0.0003. Refuted (consistent with Vlad's E-071 CE heads).
379: 
380: # Round 5 (2026-09-30 evening): maximum extractable signal
381: ## Ensemble ceiling of everything trained so far (folds 0+1, pure chains)
382: - average of all 22 head-FT members (h128/h256/distilled/gen-2/attn/two-stream/class-heads/c500/...): 0.70277,
383:   C analog (4 members) 0.70242 -> unlimited inference budget buys +0.0004 over C.
384: - average of all ~75 members (parents, full-FT, refuted arms): 0.70249.
385: - cross-fitted weights over 15 diverse members: +0.0002 over their equal-weight average.
386: C already captures nearly all the signal these model families see.
387: ## High-capacity teacher: causal segment-recurrent Transformer (`src/wnn/xl_transformer.py`, `--arch tfm`)
388: 6 layers, d 256, 8 heads, ff 1024, blocks of 250 rows attending to themselves and the previous block per layer
389: (memory carried across chunks, zero memory at sequence start), ALiBi; 4.9M parameters; chunk-invariant and causal
390: (tested). Fold 0/1 runs in progress (lr 5e-4, EMA-4 recipe, head-only FT freezing inp/blocks/ln).
391: 
392: ## Target law, classes 0-2 (subagent, 200 train sequences per class, move tables on 150, scored on 50)
393: - Targets change only at i0 price-code changes that MOVE ask or bid (>=99.98%); pure shape changes never change them.
394: - Spread term is an exact function of the 22-column price code in every class on rows with both targets <= 0
395:   (100.00% held out; tables of 53 / 490 / 3,793 / 24,307 codes for classes 3/2/1/0): refutes E-074, whose
396:   inconsistency came from floored rows.
397: - Losing side = -(sp + 2m), 1 <= m <= winner, m = clip(first adverse move of the loser's exit quote, 1, w): exact
398:   (given true F) 99.8 / 99.3 / 94.5 / 86.0% for classes 3/2/1/0 vs 99.9 / 98.5 / 77.5 / 60.7% for the fixed floor.
399: - Best F = round(mean mid over the next N ask/bid-move events): N = 4/6/6/8 for classes 3/2/1/0; exact (t0,t1)
400:   80 / 62 / 42 / 26% of rows (58 / 38 / 21 / 13% of |y| weight). F is exact for |F-mid| < 1 tick (97/86/70/52%) and
401:   fails for >= 2-tick moves in every class (<= 2-28%); no single event or row horizon fits both regimes.
402: All unexplained parts are future-dependent; the known parts (spread, loser floor shape) are visible to the model.
403: No new lever; aux targets from the future quote path were refuted earlier (E-251).
```


## E08 — Matched-seed width result

Source: `C/research/iterations/2026-09-30-creative/M01-paired-seeds.result.json`

SHA-256: `024d9b3228999e135bf71c483891bdc08d490ecb447aad58bf09604dab3cf319`


Lines 1–163:

```text
1: {
2:   "sequence_count": 860,
3:   "chain_count": 250,
4:   "matched_pairs": [
5:     {
6:       "candidate_wp": 0.6999054708402486,
7:       "reference_wp": 0.6998323760965294,
8:       "delta": 7.309474371919311e-05,
9:       "delta_by_fold": {
10:         "0": 0.00027933425887394225,
11:         "1": 0.00030299060714367165
12:       },
13:       "conditional_chain_sd": 0.0009876916825304562,
14:       "conditional_chain_ci95": [
15:         -0.0018233872822378734,
16:         0.0020593994962119135
17:       ],
18:       "bootstrap_nonpositive_fraction": 0.457,
19:       "seed": 3,
20:       "candidate": "ft256H",
21:       "reference": "ft128H"
22:     },
23:     {
24:       "candidate_wp": 0.7001665801633898,
25:       "reference_wp": 0.7005059710987173,
26:       "delta": -0.0003393909353274216,
27:       "delta_by_fold": {
28:         "0": -6.637724989067006e-05,
29:         "1": 0.0005260082812311717
30:       },
31:       "conditional_chain_sd": 0.0010019234584286538,
32:       "conditional_chain_ci95": [
33:         -0.002288161884039955,
34:         0.001489223563134759
35:       ],
36:       "bootstrap_nonpositive_fraction": 0.62,
37:       "seed": 4,
38:       "candidate": "ft256s4H",
39:       "reference": "ft128s4H"
40:     },
41:     {
42:       "candidate_wp": 0.7005845835019745,
43:       "reference_wp": 0.6997929842974587,
44:       "delta": 0.0007915992045157694,
45:       "delta_by_fold": {
46:         "0": 0.0018411619572724636,
47:         "1": 8.400613490622266e-05
48:       },
49:       "conditional_chain_sd": 0.0013609641011557846,
50:       "conditional_chain_ci95": [
51:         -0.0019839119155808907,
52:         0.003239876627439628
53:       ],
54:       "bootstrap_nonpositive_fraction": 0.274,
55:       "seed": 5,
56:       "candidate": "ft256s5H",
57:       "reference": "ft128s5H"
58:     }
59:   ],
60:   "mean_prediction_ensemble": {
61:     "candidate_wp": 0.7019357117947905,
62:     "reference_wp": 0.7012394742485428,
63:     "delta": 0.0006962375462477022,
64:     "delta_by_fold": {
65:       "0": 0.0011102284128265971,
66:       "1": 0.0006728246577072383
67:     },
68:     "conditional_chain_sd": 0.0010054620558911129,
69:     "conditional_chain_ci95": [
70:       -0.0013150890777867963,
71:       0.0026286875905561276
72:     ],
73:     "bootstrap_nonpositive_fraction": 0.239
74:   },
75:   "training_seed_summary": {
76:     "replications": 3,
77:     "matched_delta_mean": 0.00017510100430251363,
78:     "matched_delta_sample_sd": 0.000572353590016062,
79:     "matched_delta_min": -0.0003393909353274216,
80:     "matched_delta_max": 0.0007915992045157694,
81:     "positive_pairs": 2,
82:     "conditional_chain_ci95_of_mean_delta": [
83:       -0.0018869544746216095,
84:       0.002129311877716746
85:     ],
86:     "caveat": "Few reused seeds describe observed optimizer variation; no population or selection-adjusted CI is claimed."
87:   },
88:   "evidence_status": "exploratory_reused_folds",
89:   "not_estimated": [
90:     "adaptive selection bias",
91:     "hidden-test distribution shift"
92:   ],
93:   "confirmation_established": false,
94:   "provenance": {
95:     "manifest": "research/iterations/2026-09-30-creative/seed-audit.json",
96:     "manifest_sha256": "ddaa234d8f4c7925e88964c7deb1035777822b6e913de592d254271562a5bafa",
97:     "code_commit": "6d2e963270e08e166d4cd675f556ee05b266bc08",
98:     "script_sha256": "8aef289e36c2f000a56c935c2a3b5b2e8a1e7493b1091102cb6edaab9eac76c1",
99:     "scorer_sha256": "d05791f5f8c6ea64d9863d4f4b45430eb46b9d0b29ccbdac1a4ff5b36a2573c0",
100:     "python": "3.13.15",
101:     "numpy": "2.2.6",
102:     "finished_utc": "2026-09-30T21:13:27.628180+00:00",
103:     "selected_row_groups_sha256": "03ec8ec325cbe4442b0cd30301148f14eb3f1c306506f15407565db7f5eb4076",
104:     "inputs": {
105:       "cache/chain_folds.npz": {
106:         "bytes": 562791,
107:         "sha256": "315b84f385f5368615c76087288b623f99c33bd680e9ebf7c3465bf5ab8afd35"
108:       },
109:       "cache/pred_ft128H_f0_valid.npy": {
110:         "bytes": 299680128,
111:         "sha256": "7ecc594f52594a329fcb1a9f2e7efb74f320da67c6a0baa36f4406a6034a1b1e"
112:       },
113:       "cache/pred_ft128H_f1_valid.npy": {
114:         "bytes": 299680128,
115:         "sha256": "2f9727343331f9b81c604e44a1c760e6a58977ee6af991d3554d589e484f5125"
116:       },
117:       "cache/pred_ft128s4H_f0_valid.npy": {
118:         "bytes": 299680128,
119:         "sha256": "b991bd708da561c4c8adb4725a6a0cea5cb53327d9813b351c479f724b0cbe0d"
120:       },
121:       "cache/pred_ft128s4H_f1_valid.npy": {
122:         "bytes": 299680128,
123:         "sha256": "2c4beacbdde3c9be96a9c67bccb1ae1c1d61bf8e1c3f6535fa5732a295c439be"
124:       },
125:       "cache/pred_ft128s5H_f0_valid.npy": {
126:         "bytes": 299680128,
127:         "sha256": "8917f0d9f69e53ca380a1c46b7d54f9d4b4fdbdf476f7fbb1d5306086f1b8a5e"
128:       },
129:       "cache/pred_ft128s5H_f1_valid.npy": {
130:         "bytes": 299680128,
131:         "sha256": "94d757ccc93b9f065aeb408a7917dad8c5bd924f2e782b4b93d423784b7b8126"
132:       },
133:       "cache/pred_ft256H_f0_valid.npy": {
134:         "bytes": 299680128,
135:         "sha256": "0df6e50999a5d7cf61b1256744927316d5a1a03924684fd041a294abaf98d3a4"
136:       },
137:       "cache/pred_ft256H_f1_valid.npy": {
138:         "bytes": 299680128,
139:         "sha256": "62d6568d903884c1a3c9541b96fb3516cb4f148f5347c35beee64e8936545b00"
140:       },
141:       "cache/pred_ft256s4H_f0_valid.npy": {
142:         "bytes": 299680128,
143:         "sha256": "cee2d4b46993ba6ec12a61257e32d06bdd3c7bd8db92be516f601b407647401e"
144:       },
145:       "cache/pred_ft256s4H_f1_valid.npy": {
146:         "bytes": 299680128,
147:         "sha256": "6854d11ef479e32c7bc4fb00ccffbd3c07568951fe45dae17dd959dab45eb6ac"
148:       },
149:       "cache/pred_ft256s5H_f0_valid.npy": {
150:         "bytes": 299680128,
151:         "sha256": "ceae75eb7007bc1df64ca14c0123dbb07cfb49efbf20f51ed2a67015480cc063"
152:       },
153:       "cache/pred_ft256s5H_f1_valid.npy": {
154:         "bytes": 299680128,
155:         "sha256": "7bba7441d630ba32d005315e81a13210c29e3c7023d953bc681c8c1f0f5f404d"
156:       },
157:       "cache/valid_targets.npz": {
158:         "bytes": 337186198,
159:         "sha256": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955"
160:       }
161:     }
162:   }
163: }
```


## E08 — Conditional mixture coverage and score

Source: `C/research/iterations/2026-09-30-creative/I1-result.json`

SHA-256: `af02430f6cdc4383483b497f6549f20139ca458feaf9697e3a7d84e26fda3cab`


Lines 1–205:

```text
1: {
2:   "scores": {
3:     "fold_0": {
4:       "sequences": 425,
5:       "chains": 123,
6:       "wp": {
7:         "originalC": 0.715490344813047,
8:         "Q": 0.7155650192574956,
9:         "U": 0.7154184578857784,
10:         "G": 0.7155170408411597
11:       },
12:       "comparisons": {
13:         "G_minus_U": {
14:           "delta": 9.858295538134332e-05,
15:           "conditional_chain_sd": 9.064871791566015e-05,
16:           "conditional_chain_ci95": [
17:             -8.940559482495946e-05,
18:             0.00026426947669227205
19:           ]
20:         },
21:         "G_minus_originalC": {
22:           "delta": 2.66960281127826e-05,
23:           "conditional_chain_sd": 0.00010283559511406599,
24:           "conditional_chain_ci95": [
25:             -0.00017592971673708046,
26:             0.00022094376965923197
27:           ]
28:         },
29:         "G_minus_Q": {
30:           "delta": -4.7978416335858753e-05,
31:           "conditional_chain_sd": 3.041733395737792e-05,
32:           "conditional_chain_ci95": [
33:             -0.00011796647211700928,
34:             -4.35629713967934e-06
35:           ]
36:         }
37:       },
38:       "coverage": {
39:         "scored_rows": 761688,
40:         "gated_rows": 5863,
41:         "row_fraction": 0.007697377403871402,
42:         "target_weight": [
43:           529811.5765396506,
44:           525072.2864583582
45:         ],
46:         "gated_target_weight": [
47:           4028.849654749036,
48:           4491.840820968151
49:         ],
50:         "metric_weight_fraction": 0.008077373040384894,
51:         "weight_fraction_by_target": [
52:           0.007604306574542205,
53:           0.008554709392997805
54:         ]
55:       }
56:     },
57:     "fold_1": {
58:       "sequences": 435,
59:       "chains": 127,
60:       "wp": {
61:         "originalC": 0.6898641506037166,
62:         "Q": 0.6899682205338081,
63:         "U": 0.6898712057673082,
64:         "G": 0.6899589420254981
65:       },
66:       "comparisons": {
67:         "G_minus_U": {
68:           "delta": 8.773625818991881e-05,
69:           "conditional_chain_sd": 7.834343381679543e-05,
70:           "conditional_chain_ci95": [
71:             -5.478752159816802e-05,
72:             0.00025235791363855114
73:           ]
74:         },
75:         "G_minus_originalC": {
76:           "delta": 9.479142178148159e-05,
77:           "conditional_chain_sd": 0.00016653789555381757,
78:           "conditional_chain_ci95": [
79:             -0.00022731873413166237,
80:             0.00040392696296540065
81:           ]
82:         },
83:         "G_minus_Q": {
84:           "delta": -9.27850831000665e-06,
85:           "conditional_chain_sd": 1.2056763807639547e-05,
86:           "conditional_chain_ci95": [
87:             -3.444061818210775e-05,
88:             1.2960779293910697e-05
89:           ]
90:         }
91:       },
92:       "coverage": {
93:         "scored_rows": 852796,
94:         "gated_rows": 6656,
95:         "row_fraction": 0.007804914657198204,
96:         "target_weight": [
97:           578477.2378976792,
98:           594308.2509456575
99:         ],
100:         "gated_target_weight": [
101:           4841.157968237996,
102:           4579.830537125468
103:         ],
104:         "metric_weight_fraction": 0.008033002279602677,
105:         "weight_fraction_by_target": [
106:           0.008368795954412813,
107:           0.007706153380569909
108:         ]
109:       }
110:     },
111:     "pooled": {
112:       "sequences": 860,
113:       "chains": 250,
114:       "wp": {
115:         "originalC": 0.7024238010052843,
116:         "Q": 0.7025087883129355,
117:         "U": 0.702390365955736,
118:         "G": 0.7024807731272869
119:       },
120:       "comparisons": {
121:         "G_minus_U": {
122:           "delta": 9.040717155084366e-05,
123:           "conditional_chain_sd": 5.970935728955969e-05,
124:           "conditional_chain_ci95": [
125:             -2.475887929605791e-05,
126:             0.0002076192361628481
127:           ]
128:         },
129:         "G_minus_originalC": {
130:           "delta": 5.6972122002507675e-05,
131:           "conditional_chain_sd": 0.00010312637954247201,
132:           "conditional_chain_ci95": [
133:             -0.00015317262046021796,
134:             0.0002492790100743297
135:           ]
136:         },
137:         "G_minus_Q": {
138:           "delta": -2.801518564865546e-05,
139:           "conditional_chain_sd": 1.5707632574325638e-05,
140:           "conditional_chain_ci95": [
141:             -6.371039152791302e-05,
142:             -3.2541960305970713e-06
143:           ]
144:         }
145:       },
146:       "coverage": {
147:         "scored_rows": 1614484,
148:         "gated_rows": 12519,
149:         "row_fraction": 0.0077541802829882486,
150:         "target_weight": [
151:           1108288.8144373298,
152:           1119380.5374040157
153:         ],
154:         "gated_target_weight": [
155:           8870.007622987032,
156:           9071.67135809362
157:         ],
158:         "metric_weight_fraction": 0.008054013476573816,
159:         "weight_fraction_by_target": [
160:           0.008003335870073064,
161:           0.008104188928576485
162:         ]
163:       }
164:     }
165:   },
166:   "descriptive_class_deltas_pooled": {
167:     "0": {
168:       "G_minus_U": 4.752384553907696e-05,
169:       "G_minus_originalC": 5.8668597492372854e-05,
170:       "G_minus_Q": -1.4497248773337468e-05
171:     },
172:     "1": {
173:       "G_minus_U": 3.475731206903365e-05,
174:       "G_minus_originalC": 0.00010494451058606913,
175:       "G_minus_Q": -4.387833309693612e-06
176:     },
177:     "2": {
178:       "G_minus_U": 0.00019438265666338328,
179:       "G_minus_originalC": 0.0002726734094012917,
180:       "G_minus_Q": -1.9031901671628937e-05
181:     },
182:     "3": {
183:       "G_minus_U": 0.00020980663087133422,
184:       "G_minus_originalC": -2.1134663699351464e-05,
185:       "G_minus_Q": -8.43585918941514e-05
186:     }
187:   },
188:   "allocation_gate_passed": false,
189:   "confirmation_established": false,
190:   "evidence_status": "exploratory_fixed_teacher_mixture_on_reused_folds",
191:   "uncertainty_scope": "whole-chain sampling conditional on these fitted artifacts",
192:   "not_established": [
193:     "isolated conditioning effect",
194:     "student improvement",
195:     "training-seed robustness",
196:     "selection-adjusted significance",
197:     "public or hidden-test improvement"
198:   ],
199:   "provenance": {
200:     "input_inventory": "research/iterations/2026-09-30-creative/I1-result.inputs.json",
201:     "inventory_sha256": "cd50fc420aa01abea5b7851e5c0fff4aa4ed386ee971acd358adfb711eb9d13a",
202:     "selected_row_groups_sha256": "03ec8ec325cbe4442b0cd30301148f14eb3f1c306506f15407565db7f5eb4076",
203:     "finished_utc": "2026-09-30T21:23:35.794136+00:00"
204:   }
205: }
```


## E09 — Latest deep-audit synthesis

Source: `C/research/iterations/2026-10-01-deep/report.md`

SHA-256: `294f6602687ca6b92b4b73e5a8b2ccd340b1e1dda554f1b2894bd0eaac5143cd`


Lines 1–79:

```text
1: # Углублённый разбор трёх исследовательских веток
2: 
3: Предыдущий scout недостаточно отделял правдоподобное объяснение от проверенной предпосылки. В этом раунде выполнены три заранее зафиксированные CPU-диагностики на beleriand, разобраны обучение, состояние моделей и прошлые отрицательные результаты. Новых моделей, validation-WP сравнений и отправок в соревнование не было. Все доступные данные уже использовались в исследовании: это discovery, tier C, а не независимое подтверждение переноса на private leaderboard.
4: 
5: | Ветка | Что теперь измерено | Решение |
6: |---|---|---|
7: | Aux-loss при замороженной GRU | В двух ранних окнах суммарный auxiliary-gradient согласован с основным; global clipping действительно меняется | Сохранить дешёвую абляцию, ослабить объяснение через конфликт. Сравнивать с двухэпохальным V1-подобным контролем |
8: | Clock событийной ветки | На class 3 удаляется 35.67% proxy-обновлений; классы 0–2 полностью остаются на fallback | Различие реально, надёжность декодера ещё недостаточно установлена. Более чистая архитектурная альтернатива — передавать накопленное row-state в event-GRU |
9: | Разреженные изменения объёмов | Точный тест двухатомной реконструкции покрывает лишь 6.91–11.93% изменений при пороге ошибки 10%; требовалось 80% | Закрыть проверенный дешёвый линейный словарь. Не переносить его в GRU и не называть координаты физическими очередями |
10: 
11: ## 1. Обучение: отделить конфликт целей от задачи ансамбля
12: 
13: После заморозки `inp,gru` остаётся обучаемая нелинейная голова: общий Linear→GELU, затем 23 выхода. Основные два прогноза и 21 auxiliary-выход используют один скрытый слой. Поэтому снятие aux-loss действительно меняет main-представление, а не только неиспользуемые выходы. Но наличие общего слоя само по себе не означает вреда.
14: 
15: Обнаружены три разных механизма. Основной `gcorr` считает корреляцию внутри batch×chunk с весом `|clip(y)|`; auxiliary-задачи используют другие нормировки и маски без этого веса. Кроме того, градиенты всех выходов входят в общий clip-norm, включая приватные auxiliary-строки выходной матрицы. Наконец, оптимизируется корреляция отдельного участника, хотя соревнование оценивает среднее сырых прогнозов четырёх участников, обрезанное после усреднения.
16: 
17: На одном фиксированном родителе, 16 разных FT-training chains и двух первых окнах получили:
18: 
19: | Окно | Cosine main / суммарный aux на общем слое | Clip factor с aux / без aux |
20: |---|---:|---:|
21: | 0–1999, без warm-up | +0.0684 | 0.74576 / 0.80763 |
22: | 2000–3999 | +0.2943 | 0.50862 / 0.53715 |
23: 
24: Рассказ «aux тянут голову в противоположную сторону» на этих точках не подтверждается. Clipping coupling подтверждён, но изменение его коэффициента не равно изменению шага Adam на тот же процент: адаптивные моменты частично компенсируют общий масштаб. Диагностика не воспроизводит двухэпохальную траекторию и не определяет, какой loss даст лучший score. Предсказаний за пределами ±2 в этих окнах не было, поэтому немедленное объяснение через clipping tails также не получено.
25: 
26: Более содержательная резервная идея — **обучать выбранную голову для фиксированного ансамбля**. Корреляция отдельной модели почти не меняется от положительного масштабирования её прогноза. Но внутри среднего из четырёх моделей масштаб одного участника меняет его фактический вес. Можно заморозить остальные три прогноза и оптимизировать корреляцию их суммы с обучаемым участником. Для контроля нужен и дешёвый affine-вариант: четыре параметра масштаба/сдвига, по два на target. Если он даёт тот же эффект, улучшена калибровка ансамбля, а не нелинейное представление. Это отдельная гипотеза; она ещё не проверена и может переобучиться на in-sample ошибки партнёров.
27: 
28: Следующая простая абляция всё ещё разумна из-за малой стоимости и отсутствия overhead на serving. Однако предыдущий план сравнивал одноэпохальные головы C, тогда как сохранённый public-best V1 обучает головы две эпохи **заново от родителя**, с другим OneCycle schedule. Корректный минимальный V1-подобный протокол теперь содержит 12 head fits: четыре candidate/control пары для plain-h128 seeds 3/4 × folds 0/1, плюс четыре двухэпохальных контрольных якоря для distilled-h256 и chunk500. Родители не переобучаются; финальный checkpoint epoch1, веса .25, один aux-loss multiplier 0 против 1. Конструктор auxiliary-голов и размеры checkpoint остаются прежними.
29: 
30: Ориентир из существующих логов — 15–25 GPU-минут плюс prediction/startup/I/O; это оценка, не новый benchmark. Основной readout — замена двух plain-slots в одном и том же двухэпохальном ансамбле, отдельно каждый seed/fold и pooled. Предложенный порог дальнейших затрат: +0.0003 pooled и положительный эффект в обоих фолдах для замены двух участников. Каждый standalone и отдельный slot effect обязательно публикуется; их неоднородность требует осторожности и повторения, но ухудшение standalone не является автоматическим запретом на полезный ансамбль. Порог не является статистической гарантией и перед запуском должен быть частью окончательного training contract.
31: 
32: Полный разбор формул, Adam/OneCycle, масок, происхождения auxiliary-меток, прошлых E-097/E-102/E-148/E-171 и точных контролей: [training.md](training.md). Измерения: [training-result.json](training-result.json).
33: 
34: ## 2. Архитектура: содержимое событийной памяти важнее названия clock
35: 
36: Текущий clock — одновременное изменение первых закодированных bid/ask координат. Event-GRU получает моментальную проекцию строки `z_t` и log-gap, но не накопленное состояние обычной row-GRU `h_t`. Следовательно, раздельные часы сами по себе ещё не обеспечивают сохранение того, что происходило между событиями.
37: 
38: Новый lookup обучен на 96 class-3 train-only chains; проверка — на 32 других train-only chains, по восемь на класс, без чтения их targets. На class 3:
39: 
40: - lookup знает 32,657 из 32,663 изменений кодов — 99.98%;
41: - 11,648 из 32,655 proxy events оказываются нулевыми движениями в прямом декодере — 35.67%;
42: - пропущенных proxy-clock ненулевых декодированных движений не найдено;
43: - прямой clock и clock реконструкции через спред расходятся лишь на четырёх известных событиях — 0.0123%;
44: - prefix, chunk и streaming проверки прошли.
45: 
46: Но известный ключ не равен надёжно установленному физическому движению. Только 56.98% событий удовлетворяют заранее выбранным требованиям support/purity. Кроме того, 44 из 237 общих ключей свежей и исторической таблиц расходятся по signed delta. Это **не** означает 18.57% ошибочных событий или binary-clock ошибок: различается единица подсчёта и сам проверяемый объект. Зарегистрированный semantic gate всё же не пройден; применять декодер как доказанную истину рано. На классах 0–2 ни одного известного перехода не оказалось, там сохраняется старый clock.
47: 
48: Более чистая следующая архитектурная гипотеза: при прежних событиях подавать в event-GRU **`h_t` вместо `z_t`**. Размеры, параметры, количество операций и event-state относительно текущей DualClock одинаковы; меняется информация в токене. Это также меняет путь градиента через row-GRU, поэтому положительный результат нельзя автоматически приписать только лучшему запоминанию интервалов. Проверять нужно против той же DualClock с `z_t`, обычной GRU и фиксированной замены участника ансамбля. Предыдущие отрицательные dual-clock и event-attention эксперименты существенно понижают prior, но именно эту замену не тестировали.
49: 
50: Важная найденная ловушка: `freeze=inp,gru` не замораживает `event_gru`. Для настоящего head-only сравнения требуется явно заморозить и её; у attention аналогично остаются дополнительные обучаемые модули. Старые результаты сохраняются, но название этапа не гарантирует одинаковую глубину fine-tune разных архитектур.
51: 
52: Стоимость также не нулевая относительно обычной GRU: при H=E=128 дополнительные MACs составляют примерно `16,384 + 98,688 × event_rate` на строку. Native DualClock пока нет. Поэтому нужны измерение полного replacement-ансамбля и причинная parity-проверка; разницу между наблюдаемыми 41m42s V1 и лимитом 60 минут нельзя считать гарантированным запасом. Отдельный слабый резерв — зависимость readout событийного состояния от его возраста, без изменения записи; не смешивать её с заменой токена или clock в одном эксперименте.
53: 
54: Полный разбор реализации, прошлых E-056/R6/attention, декодера и стоимости: [architecture.md](architecture.md). Измерения: [architecture-result.json](architecture-output/architecture-result.json).
55: 
56: ## 3. Quant: физические очереди и полезные координаты — разные утверждения
57: 
58: Идея словаря имела конкретную предпосылку: если коды являются фиксированным линейным смешиванием объёмов, изменение одной-двух очередей лежит в пространстве одной-двух колонок матрицы. При нелинейной анонимизации направления зависят от текущего состояния. При нескольких обновлениях между строками наблюдается только net change, по которому нельзя посчитать отдельные поступления и снятия. Перестановка, знак, масштаб и физический ноль также не восстанавливаются из хорошей реконструкции сами собой.
59: 
60: Проверены 64 разные train chains, две половины без пересечения цепочек. Только features, без labels/predictions. Фиксированный 11-направленный словарь сравнивался с исходными осями, PCA и случайным базисом, при одинаковой двухатомной реконструкции и весе каждой последовательности.
61: 
62: | Сторона и направление fit→audit | Доля изменений с относительной ошибкой ≤10% |
63: |---|---:|
64: | Bid A→B / B→A | 6.91% / 7.29% |
65: | Ask A→B / B→A | 8.97% / 11.93% |
66: 
67: Требовалось 80%. Медианная ошибка — 24–28%; ask-базисы плохо обусловлены и менее стабильны. Sentinel-исключений в выборке не было. Результат не объясняется одним неудачным классом: ни один из заранее предусмотренных классов не приближается к gate. Положительный контроль на искусственном фиксированном ортогональном смешивании успешно восстанавливает данные с одним и двумя изменениями; он проверяет работоспособность простого случая, но не доказывает полноту алгоритма на произвольных словарях. Fit оптимизирует ближайшую линию, а gate оценивает две линии; это остаётся ограничением эвристики.
68: 
69: Практическое решение: этот дешёвый линейный декодер закрыт, обучение GRU на его координатах не оправдано. Нельзя объявлять все quant-методы или нелинейные представления бесполезными. Нельзя и автоматически увеличивать словарь/менять преобразования до получения приятного reconstruction-score.
70: 
71: Более обоснованный quant-резерв — **условная неожиданность trade-code события**. Из прошлых наблюдаемых категорий предсказывается распределение следующей категории; после её прихода вычисляется `onehot(mark) − P(mark | history)`. Это не требует знать физическую сторону сделки. Смысл — отличить ожидаемый устойчивый поток от неожиданного события. Сначала нужна причинная проверка prequential log-loss против постоянных частот; затем сравнение с raw mark/frequency признаками той же мощности внутри фиксированного price-предиктора. Хороший прогноз категории сам по себе не означает полезного прогноза цены. Это резерв, не выполненный эксперимент; старые отрицательные trade-flow результаты остаются существенным контраргументом.
72: 
73: Полный разбор идентифицируемости, требований queue-reactive литературы, отрицательных E-090/E-215/E-112 и возможных контролей: [quant.md](quant.md). Измерения: [quant-result.json](quant-result.json), [synthetic positive control](quant-toy-result.json).
74: 
75: ## Что теперь делать в первую очередь
76: 
77: Первым остаётся ограниченный двухэпохальный aux-off/control эксперимент: он дешёвый и не увеличивает serving, но механизм конфликта больше не выдаётся за установленный. Следующая содержательная training-альтернатива — ensemble-aware head FT с affine-контролем. Архитектурный приоритет смещается от неподтверждённого «исправления clock» к конкретному `z_t → h_t` в событийной ветке. Quant-словарь не продвигается; условная неожиданность потока заслуживает только своего дешёвого причинного preflight.
78: 
79: Три CPU-процесса заняли суммарно около 36 секунд wall time после подготовки; каждый использовал не более двух потоков и около 1.0–1.2 GiB RSS. Это затраты диагностик, не всего исследования. Исходный server checkout, trained checkpoints, evaluator и прошлые результаты не менялись. Новая SOTA этим раундом не получена.
```


## E10 — Reframer: leader rejection and row-thinning screen

Source: `R/campaign_2026-10-01/creative/cr_reframer/REPORT.md`

SHA-256: `22b5c8008aa2e75af48d18f03f41027fc091b76652d7d21b25857a260302cd7a`


Lines 1–23:

```text
1: # cr_reframer — reframing cards and probes (final, saved by controller from the agent's message)
2: 
3: Reference: OOS ft_since s3+s4 mix, folds 0–3, pooled WP 0.68952. Cards committed 20:29 before any probe (cards.json);
4: R11/R12 added post hoc. Numbers in results.json; scripts scripts/cr_reframer_*.py.
5: 
6: | card | lens | outcome | status |
7: |---|---|---|---|
8: | R02 per-sequence level/scale oracles | Factor | in-sample +0.0289 / +0.0141; episode cross-fit −0.021 / −0.012; best shrinkage +0.0008 / +0.0005; first-half→second 0.0000 | supported (negative): oracle headroom ≈97% overfit |
9: | R01 i1 as second labelled stream | Change search unit | class vocabularies disjoint; i1 overlaps own-class i0 vocab 29/51/41/12%; no i1 state = another sequence's i0 state; 20.6% unknown i1 transitions in class 3 | screen_failed |
10: | R05 i1 as leader | Instrument | class 3: i1 0.089 code events/row vs i0 0.309 — slower separate instrument | screen_failed |
11: | R03 censored volume codes | Re-encode | ~10^5 distinct values/column, no max pile-up | screen_failed |
12: | R12 decoded level reversion / i0–i1 error correction (post hoc) | Instrument | partial corr ±0.04; class-3 LOFO correction +0.00001 ± 0.00017, folds disagree | screen_failed |
13: | R06 softer public mask | Relax | p²+0.03 weights already put 11.7% of mass on prop<0.01 rows (λ=0.05 asks 12.5%); LB gate probe would only settle the proxy | screen_failed (training lever) |
14: | R04 event-time multi-horizon labels | Change time | first event resolves 15–22% of D variance; next episode keeps 56–76% | screen_failed |
15: | R11 event-preserving row thinning (post hoc) | Change time | dropping 21% of rows with no i0 price-code change keeps every kept label exact; untrained incumbent 0.7445→0.7438 on 80 seqs (c2 −0.0046, c3 +0.0027), pred corr 0.995–0.998; view average +0.0002 | insufficient_evidence |
16: | R07–R10 | Mine surprises/Transfer/Invert/Domain op | prior evidence already below falsifiers | not probed |
17: 
18: Where not to dig: sequence-level latents (FiLM, context encoders, online recalibration); price-level or i1 information;
19: the scorer mask (weights already match). Binding constraint: estimation variance on ~28 effective episodes per class-3 sequence.
20: Proposed 2-GPU-h experiment: R11 row-thinning augmentation (loader-only: per sequence w.p. 0.5 drop step≥99 rows with
21: unchanged i0 price codes at rate q~U(0.1,0.4); thin x/y/mask/propensity; recompute event features and aux labels; pad with
22: zero-weight rows; predict unthinned). Arms seeds 3,4 × folds 0,1, 4 epochs; gate pair Δ ≥ +0.0008, both folds > 0,
23: class 3 ≥ −0.0003; kill Δ ≤ 0 or class 2 < −0.002. Expected +0.0002…+0.001, P(pass) ≈ 0.2.
```


## E10 — Residual diagnostics and limits of falsifier family

Source: `R/campaign_2026-10-01/creative/cr_surprise/REPORT.md`

SHA-256: `c96c1a3d4ca1e0f3cf4001d0ff5dadef5d5f8ecf1f9b7d6df41759f9a9215552`


Lines 1–19:

```text
1: # cr_surprise — residual forensics (final, saved by controller from the agent's message)
2: 
3: Basis: clean OOF since-FT pair, all 1873 valid sequences, Global WP 0.6895. Class 3 = 47.7% of weight, 60.5% of 1−WP.
4: Numbers: results.json; figure event_time.png; falsifiers hazard/tier/rawprobe_falsifier.json.
5: Verdict: loss is concentrated, but 18 cross-fitted remaps and 5 falsifiers all score ≤ 0 — no causally observable fix.
6: 
7: | # | surprise | at stake | falsifier result |
8: |---|---|---|---|
9: | 1 | Information arrives late inside a run: corr(run target, p at run start) 0.105 in class 3 vs 0.81 at run end (class 0: 0.70 vs 0.87). Class-3 rows ≥32 before next E = 50% of class-3 weight, 72% of its loss, WP 0.02–0.33. | non-causal until_E remap +0.048 class-3 | causal hazard predictable (R² 0.61) but class×hazard-decile remap −0.0012: model already prices hazard |
10: | 2 | Per-sequence trend level in class 3 (seq corr of D level 0.53; sd of seq-mean D 0.43 y vs 0.15 p) | oracle +0.029 global | causal drift (20–8000 rows) |corr| ≤ 0.025; linear augmentation −0.004 class 3 |
11: | 3 | Class-3 tier: same p for (0,−1) and (+1,−3) — direction found, not whether price moves through; winner rows = 86% of class-3 loss | — | AUC 0.666 p only / 0.679 summaries / 0.660 raw 112 features; raw-book residual MLP −0.011 |
12: | 4 | 29% of class-3 Var(p) is within-run, zero covariance | run-mean oracle +0.025 | causal mean −0.019; F-space/y-space EMA lose |
13: | 5 | E-row over-shift (slope 1.37 vs 1.025) | ≤ +0.0001 | fold-inconsistent |
14: | 6 | Mask avoids quote-move bursts in class 3 (3% scored on E rows vs 21% quiet) | — | propensity cache already reproduces it |
15: | 7 | Class-3 per-sequence S level anti-correlated with truth (−0.14) | oracle +0.0044 class 3 | S-shift cross-fit ≤ 0 |
16: 
17: Also: seed disagreement carries no confidence signal; parent beats FT in class 3 at since_E 32–512 (already used by RSP2 routing).
18: Where to dig: history-based decoding of the class-3 queue race (which queue depletes first; does price move through) early
19: in quiet runs. Volume codes are mixed, not depth-ordered; every post-hoc/summary-feature route is exhausted.
```


## E11 — Campaign chronology, public composition assumptions and forecasts

Source: `R/campaign_2026-10-01/REPORT.md`

SHA-256: `c56f7f4654b692bd8e6c2070033878aa1fcab4cd1c75601ca8a541eb2e4ff7b7`


Lines 1–135:

```text
1: # Overnight research report, 2026-10-01 (02:10 → 10:00 Kyiv)
2: 
3: Status: FINAL (written 18:05; research cutoff 10:00 respected — no job launched after 06:26, all jobs finished by 06:37).
4: Three usage-limit pauses (≈03:15–04:30, ≈05:15–09:30, ≈09:35–18:00) cost most of the night's wall time.
5: Nothing was uploaded.
6: 
7: ## Starting point
8: - Our public incumbent 0.6622 (`final_since2_gclp`, 2×GRU128 since + GCLP). Team best (teammates) 0.6641.
9: - Clean 4-fold since/since reference (chain-excluded, fixed epochs): Global WP 0.689523; class WP
10:   0.807 / 0.776 / 0.739 / 0.433.
11: - Teammate ideas/numbers: `TEAM_INTAKE.md` (clean-room: ideas only, no code read by implementers).
12: 
13: ## Evidence log (E-ids in research/experiments.jsonl)
14: | id | experiment | result | verdict |
15: |---|---|---|---|
16: | E-287 | current-code spread labels vs matched proxy control (fold 0) | +0.00002 ± 0.00008 | reject |
17: | E-288 | cross-fitted per-class output scale (4 folds) | mean −0.00006, in-sample bound +0.00006 | reject |
18: | E-289 | distilled GRU256 since student, fold 0 | head-only FT single +0.00067 vs since s3 (class 3 +0.0043); pair+d256 thirds +0.00043 ± 0.00054; two-stage FT −0.0019 single | head-only; fold 1 running |
19: | E-290 | FT pair + own pre-FT parents (output mix, 4 members) | folds 0+1 +0.00097 ± 0.00043 (both folds +); holdout +0.00046 ± 0.00040 | promising, f2/f3 confirm prereg |
20: | E-291 | pooled-memory FT loss (N2) | −0.0015 two-stage / −0.0004 head | reject |
21: | E-291 | head-only FT pair vs two-stage pair | +0.00064 ± 0.00043 (holdout +0.00017) | head-only ≥ two-stage |
22: | E-291 | shared trunk: parents + head-only FT heads (2 trunks) | +0.00098 vs FT pair, = 4-trunk mix; holdout +0.00038 | candidate core |
23: | E-292 | cross-fitted weights over parent/head members | −0.00038 ± 0.00017 vs equal | reject; equal weights |
24: | E-293 | joint ensemble head FT vs matched solo (N1) | −0.00010 ± 0.00003 | reject |
25: | E-294 | SP2 + chunk-500 / online-norm merged trunks (holdout) | −0.00004 / +0.00001 | reject extra base trunks |
26: | E-295 | causal EMA smoothing / sharpening of outputs | smoothing −0.001…−0.048; sharpening ±0.00001 | reject |
27: | E-306 | class-routed heads (class 3 → parent+FT heads, classes 0–2 → two-stage FT pair), folds 0+1 | +0.0012 ± 0.0006 vs FT pair; classes 0–2 unchanged | candidate (post hoc) |
28: | E-307 | **PREREGISTERED folds 2+3 confirmation** | SP2 −0.00008 ± 0.00041 (fail); mix4 +0.00023 (fail); RSP2 +0.00031 ± 0.00033, both folds + (below +0.0005 gate) | SP2 not confirmed; RSP2 directional only |
29: | E-308 | holdout: seed bag SP3/SP4 (new final seeds 5, 6) and routed | SP4−SP2 +0.00005; new seeds weak (−0.0006); RSP2−incumbent +0.00125 ± 0.00051 | SP4 dropped; RSP2 best |
30: | E-309 | extra heads (FT-seed bag, distilled head, train+valid gcorr head) | +0.00008 / 0.0000 / −0.0002 | reject |
31: | E-310 | fp32 packages SP2_f32, RSP2_f32 | all packaging gates pass | ready, not uploaded |
32: | E-311 | distilled GRU256 fold 1 (scored after cutoff from finished run); routed RSP2_D vs RSP2 | folds 0+1 +0.00053 ± 0.00022 (fold 1 +0.00087; pub mix +0.0006) | strongest next step; final d256 not trained |
33: 
34: ## Data findings
35: - Local/public gap is composition: within-class WP is nearly identical for isolated (len-1 chain) and long-chain
36:   sequences (class 3: 0.423 vs 0.435); isolated sequences are 66% class 3 and score 0.578 pooled.
37: - Class 3 = 37% of valid sequences, 46–48% of metric weight, WP 0.433; 9 distinct target values, ~35% zeros.
38: - Implied public class mix (data_eval/class_mix.json): the clean reference (0.6895) maps to the incumbent's 0.6622 when
39:   class-3 sequence share is 0.459 (95% [0.41, 0.51]); holdout-package fit gives 0.477. Mix [0.183, 0.160, 0.198, 0.459]
40:   ("pub143") is used to weight local deltas, not to forecast absolute scores.
41: - Target laws (TRAIN, component-disjoint fit/held-out, E-302/303): targets are piecewise constant between `a6`
42:   lower-endpoint quote-move markers — they change on exactly those rows (100.000% of weight, no unknown mass);
43:   interior t0+t1 = −spread(code) exactly on known codes; loser ≈ −(s + 2·clip(m,1,W)) only partially (72/83/98/99.8%
44:   by class); teammate's "winner = rounded mean mid over next N events" rejected (2–53%). ~20% of metric-weighted
45:   variance is fixed by the current code; the rest is future path (mostly direction D, 80%). A structural
46:   p = s/2 ± d head with s from codes loses (−0.001…−0.04); the model already uses spread.
47: - Transfer slope public/local for clean chain-excluded deltas ≈ 0.8 (range 0.63–0.92; not identified for new families).
48: - Optimal-predictor check: for weights |y| the weighted-Pearson-optimal p(x) is affine in E[|y|y|x]/E[|y||x], the
49:   minimiser of |y|-weighted MSE — so the wMSE parent objective is aligned; gains must come from variance reduction
50:   (ensembles, heads, distillation) or information, not from objective tricks (consistent with N1/N2 rejections).
51: 
52: ## Candidates
53: - Merge trick: k heads on one bit-identical trunk average exactly into one head of width k·H
54:   (`scripts/package_submission.py` member syntax `PARENT@ep3|HEADFT@ep0`; torch check max diff 1e-6).
55: - Core candidate SP2: incumbent trunks `ab67_final_since_s{3,4}` each with merged head (parent head + gcorr
56:   head-only FT head). Same trunk cost as the 0.6622 incumbent.
57: - SP2+D: SP2 + final distilled GRU256 trunk with merged head (parent + head-only FT). Needs fp16 engine.
58: - SP4: four since GRU128 trunks (seeds 3, 4 incumbent + new final seeds 5, 6), each with merged head
59:   (Kaggle JS-6th seed-bag idea, made cheap by the int16 engine).
60: 
61: ## Packaged and verified
62: - `submissions/c20261001_SP2/submission.zip` — SHA-256 `1e5c6914d892b5dbffe95f4290c669c24c22f0bb30b34123daf610a24239742e`,
63:   3,147,548 bytes, 2 merged-head members (coef 0.5 each), governor off, int16. Checks: offline Python 3.11.16
64:   scorer image PASS; v2/avx2/v3 bitwise equal; A→B→A reset + warm-up contract PASS; repeat-deterministic;
65:   harness runtime 0.634× incumbent → ~23 public min. Manifest: `packages/SP2_manifest.json`. Not uploaded.
66: 
67: ## Inference engine (own Rust code, engine/results.json)
68: - Int16 weights with fp32 per-row scales: parity vs fp32 max 8.2e-5 (gate 1e-4), |ΔWP| ≤ 4e-7, bitwise identical
69:   across v2/avx2/v3; fp16 is faster but fails the 1e-4 gate (3.9e-4–5.8e-4, ΔWP ~1e-7). Head width generic.
70: - Projected public minutes (int16, calibrated on the incumbent's 36.45 min): 2×h128 19.7, 6×h128 57.6,
71:   2×h128+h256 51.5, 3×h128+h256 60.5. Budget gate 52 min. One h256 ≈ 3.5–4 h128 trunks.
72: 
73: ## Update 23:30 — new best verified candidate
74: - `submissions/c20261001_S4D_i16/submission.zip` (SHA-256 `ab0f5287…fec00`, 6.18 MB, int16, ~48 public min):
75:   since s4 trunk + final distilled GRU256 trunk, each with merged parent+head-FT heads, equal weights.
76:   Evidence: 4 clean folds +0.0018 ± 0.0005 vs the incumbent-recipe pair (S3 variant +0.0013); holdout +0.0016 ± 0.0007.
77:   d256 confirmed on preregistered folds 2/3 (SP2_D − SP2 +0.00099 ± 0.00018, both folds +). Forecast ≈ 0.6630–0.6632
78:   (P > 0.6641 ≈ 0.10–0.15; P ≥ 0.6660 < 0.001). Supersedes RSP2 as the recommended upload.
79: 
80: ## Final recommendation (earlier, 10:00)
81: 1. **Primary: `submissions/c20261001_RSP2_f32/submission.zip`** — SHA-256
82:    `2783eea96664e2f50b9403bbda610a772d4df182dc0dd6d59727898a1a8bdc20`, 6,050,069 bytes, fp32, governor off.
83:    Market-class routed: a warm-up detector (incumbent parent's auxiliary class logits, rows 50–98; 100% accurate
84:    out of fold on 925 sequences) picks, at row 99, either the incumbent's two GCLP members (classes 0–2 — identical
85:    outputs to the 0.6622 package) or, for class 3, the same two since trunks with merged heads averaging the
86:    pre-FT parent head and the head-only gcorr FT head. Evidence: holdout +0.00125 ± 0.00051 vs incumbent (+0.0015
87:    under the implied public class mix), folds 0+1 +0.0012 (post hoc), preregistered folds 2+3 +0.00031 ± 0.00033
88:    (both positive, below the +0.0005 gate). Runtime ≈ 0.79× incumbent in the same harness → ~29 public min.
89: 2. Secondary (not recommended): `submissions/c20261001_SP2_f32/submission.zip` (SHA-256 `0126c381…0ede`) — failed
90:    its preregistered confirmation; class 1/2 losses.
91: 3. Reference packages kept: `c20261001_SP2` (int16 build, SHA `1e5c6914…742e`), `c20261001_RSP2` (int16 build).
92: 
93: Gates for RSP2_f32: ZIP 6.05 MB ≤ 20 MB; root solution.py; Python 3.11.16 offline image (no network, 1 CPU, 16 GB)
94: bitwise equal to host; v3 = avx2 bitwise, v2 within 4.8e-7; A→B→A reset, warm-up None, finite outputs, fresh
95: instance identical; fixed members and weights (no governor); runtime ratio 0.79 vs the incumbent's 36.45 min.
96: 
97: ## Expected public score
98: Independent audit v1 (critic/expectation_v1.md; author ≠ auditor), 50/50 holdout route (slope 0.657 on 6 adjacent
99: public GRU steps) and fold route (pub143 delta, empirical-Bayes selection shrinkage, τ ~ N(0.75, 0.35), LB noise 0.0006):
100: 
101: | candidate | mean | 80% PI | P(>0.6622) | P(>0.6641) | P(≥0.6660) |
102: |---|---:|---|---:|---:|---:|
103: | SP2 (packaged) | 0.6626 | 0.6617–0.6635 | 0.71 | 0.02 | <0.001 |
104: | SP4 (pending seeds 5/6) | 0.6632 | 0.6621–0.6642 | 0.88 | 0.12 | 0.001 |
105: | SP2+D (runtime 51.5–57 min, not shippable in i16) | 0.6628 | 0.6618–0.6639 | 0.76 | 0.06 | <0.001 |
106: 
107: Update rule after the preregistered folds-2/3 confirmation: E[SP2] = 0.66232 + 0.375·m with m the precision-weighted
108: combination of prior N(0.0003, 0.0006²) and the f2+f3 pub143 delta (fold 0/1 evidence excluded as post hoc).
109: **0.6660 is not supported by any candidate**; it needs ≈ +0.0038, about 5× the largest selection-corrected gain.
110: 
111: ### Updated after the confirmation (author-computed with the audit-v1 method; the independent auditor hit its usage limit)
112: | candidate | mean | 80% PI | P(>0.6622) | P(>0.6641) | P(≥0.6660) |
113: |---|---:|---|---:|---:|---:|
114: | **RSP2_f32** | **0.6627** | 0.6617–0.6637 | 0.76 | 0.04 | ≈0 |
115: | SP2_f32 | 0.6624 | 0.6615–0.6633 | 0.60 | 0.01 | ≈0 |
116: | SP4 | dropped (holdout increment +0.00005; new seeds weak) | | | | |
117: 
118: Method: 50/50 mixture of (a) holdout route, public Δ = 0.657 × holdout Δ (slope from 6 submitted adjacent GRU
119: steps, jackknife SD 0.24, residual 0.00064) and (b) fold route using only the preregistered folds 2+3 delta under the
120: implied public class mix, shrunk with prior N(0.0003, 0.0006²), transfer τ ~ N(0.75, 0.35), LB noise 0.0006.
121: **Honest verdict: no new public best is established; expected ≈ 0.6627, roughly +0.0005 over our 0.6622 and below the
122: team's 0.6641. The 0.6660 target is not reached and no evidence supports it.**
123: 
124: ## How the gain was found (what worked)
125: - Data finding first: local↔public gap = market-class composition (public ≈ 46% class-3 sequences vs 37% local),
126:   so class-3 changes carry more public weight than local pooled deltas suggest.
127: - Combination of existing pieces: the pre-FT parent head and the head-only GCLP head on the same trunk are
128:   differently-biased views (parent better on class 3, FT better on classes 0–2). Averaging them exactly on one trunk
129:   (merged wide head) is free at inference; routing by causally detected class keeps the FT advantage on classes 0–2.
130: - Engineering: own Rust engine with int16 weights and generic head width (built in the scorer base image) cut a
131:   2-trunk ensemble to ~0.6× of the incumbent's runtime, making ensembles of 4 h128 trunks affordable (not needed).
132: 
133: ## Negative results and open questions
134: Rejected tonight: spread-label aux (E-287), class-scale correction (E-288), pooled-memory correlation loss (E-291),
135: cross-fitted weights (E-292), joint ensemble head FT (E-293), extra chunk-500/online-norm trunks (E-294), output
```


## E12 — Training target uses future resolution time

Source: `R/hypotheses.jsonl`

SHA-256: `8718314837bbbe1aca8c1cbbf679e17a1c7cd896272fd03d87fdd2c88be70eab`


Lines 59–59:

```text
59: {"id": "H-059", "committed_at": "2026-10-02", "mechanism_family": "noise-adaptive distillation (training-target design)", "statement": "Distillation weight should grow with the label's distance to resolution: target = (1-a(u)) y + a(u) teacher with u = rows until the next target change (known at train time from y; inputs stay causal). Expected to beat a constant alpha because the alpha 0.7 gain is concentrated far from the next quote move (F5) where ~44-50% of the metric loss lives (F3).", "control": "constant alpha 0.7 student, same teachers/seed/fold", "falsifier": "<= +0.0005 vs constant alpha on folds 0/1", "status": "proposed", "evidence": ["E-320", "E-323"]}
```


## E13 — Completed fixed-weight Transformer comparisons

Source: `C/research/iterations/2026-10-01-next/report.md`

SHA-256: `047e09c40f67b13604c156a5d5eff49b592095a565b45f953d5bc72f094be44b`


Lines 1–54:

```text
1: # Cache cleanup and next quant research directions
2: 
3: The next research priority is **main-only head fine-tuning with an unchanged frozen GRU**. A corrected quote-event clock is the next architecture preflight; decoded queue-update dynamics is the next quant representation preflight. These are proposals, not measured improvements. The completed Transformer did not pass either of the two registered ensemble-contribution gates.
4: 
5: ## Completed Transformer screen
6: 
7: The [contract](contract.json) fixed two comparisons before new scores: add the head-only or full-FT Transformer to C at weight 0.2, scaling C by 0.8. This is equal averaging of five raw member predictions before clipping. No weights, epochs, replacement arms or new seed subsets were searched.
8: 
9: | Fixed teacher mixture minus C | Fold 0 | Fold 1 | Pooled WP delta | Conditional chain 95% interval |
10: |---|---:|---:|---:|---|
11: | 0.8 C + 0.2 Transformer head FT | +0.000281 | −0.000049 | +0.000115 | [−0.000319, +0.000543] |
12: | 0.8 C + 0.2 Transformer full FT | +0.000395 | −0.000098 | +0.000126 | [−0.000372, +0.000600] |
13: 
14: C scored **0.702423801** on 860 pure validation sequences from folds 0/1. Both mixtures miss the registered +0.0005 spending threshold and have a negative fold-1 effect. Decision: **screen_failed for both fixed mixtures; no Transformer-student fit allocated from this screen**. This does not establish equivalence, a universal model ceiling or failure of every Transformer design.
15: 
16: Both evaluations used already saved predictions, two numerical threads, CPU quota 160%, an 8 GiB memory cap and a 480-second per-run limit. They completed in **15.80 seconds total**, with maximum RSS about **0.625 GiB**, and no GPU work. See the [head result](TFM-H.result.json), [full-FT result](TFM-F.result.json), [frozen input hashes and provenance](TFM-inputs.json), and [post-run identity check](TFM-postcheck.json). The new comparisons are discovery results on exposed folds; their chain intervals condition on one fitted Transformer seed and do not correct selection or estimate training variation.
17: 
18: ## Ranked next interventions
19: 
20: ### 1. Training: remove auxiliary objectives only during frozen-trunk head FT
21: 
22: `fold_ft.sh` keeps structure and next-change losses at .3/.3 even while `inp,gru` are frozen. The main outputs and auxiliary outputs still share the first nonlinear layer of the head. Those tasks can regularize the readout, but they can also compete with the final correlation objective. Existing head/full FT comparisons did not isolate this choice.
23: 
24: The next bounded training comparison should keep the trained parent, model shape, initialization, masks, data order, scheduler and one-epoch head schedule fixed. Introduce a separate auxiliary **loss multiplier** of zero versus one; setting existing auxiliary flags to zero changes checkpoint dimensions and is not the same experiment. Start from matched seeds 3/4 and assess replacement of the corresponding h128 slots in C, not just standalone WP. A training-side gradient audit can explain priorities, but a cosine threshold is not a universal veto on the intervention. Auxiliary tasks may remain helpful even when some gradients conflict. [Du et al.](https://arxiv.org/abs/1812.02224) motivate that diagnostic, not a claimed gain here.
25: 
26: This has the best immediate cost profile: parent training is reused and serving shape, state and operation count are unchanged. No new head training has been launched in this round. Exact controls, thresholds, caveats and the proposed bounded audit are in [architecture-scout.json](architecture-scout.json).
27: 
28: ### 2. Architecture: change the event branch's clock to actual decoded quote moves
29: 
30: The current dual-clock and event-attention models consume `EV_I0PX`, defined as **both first encoded bid and ask coordinates changing**. This is a proxy event predicate, not an explicit test that decoded best bid or ask moved. A train-fitted transition table can supply observed quote deltas where known; unknown edges must use a fixed fallback and remain counted.
31: 
32: First measure decoder coverage and disagreements between the two clocks, using a train-only table with verified provenance. Use direct current transition deltas; do not use a full-sequence anchoring statistic, evaluation-label class routing or future labels. Only if the clocks differ materially should one matched dual-clock intervention change the update mask while retaining the row branch, state size and recipe. It must beat both its proxy-clock control and the fixed C replacement. The previous dual-clock and event-attention negatives make this a conditional, lower-prior direction—not an excuse for an architecture sweep. Native inference/lookup cost also needs measurement.
33: 
34: ### 3. Quant features: identify queue updates before modeling depletion/replenishment
35: 
36: Two books with similar current state can behave differently when one queue is replenished and another persistently depleted. Raw cumulative volume-code changes already failed here. A distinct first question is whether **increments** of anonymized volume codes admit a stable sparse decomposition into update coordinates during unchanged-price intervals.
37: 
38: Fit one train-only dictionary and test reconstruction/stability on other train chains before using target scores. Passing would justify checking whether signs and levels can be identified; reconstruction alone does not identify physical add/remove events or queue size. Only after that separate identification would a fixed state-by-add/remove-rate screen compare with C **and** a capacity-matched raw-code control. Price-changing or undecodable increments remain outside the dictionary's validated domain. An unnamed-coordinate predictive feature needs a separate hypothesis rather than a post-failure reinterpretation. [Queue-reactive Hawkes models](https://arxiv.org/abs/1901.08938) motivate combining book state with flow history; they do not prove that these transformed features expose their required variables. The detailed limits, independent review correction and cheaper falsifier are in [quant-scout.json](quant-scout.json).
39: 
40: A lower-priority quant reserve is conditional trade-mark surprise: predict an observed trade-code mark from prior flow, then use its innovation after arrival. This differs from the already failed MO0/MO1 sums, but must beat a raw mark/frequency control; mark predictability alone does not establish price predictability. No such screen ran this round.
41: 
42: ## Metric source resolved
43: 
44: The current indexed organizer [data overview](https://wundernn.io/connectome/docs/data_overview) and [FAQ](https://wundernn.io/connectome/docs/faq) explicitly specify global pooled WP. This resolves the preceding iteration's source-access uncertainty without changing the frozen evaluator. [METRIC.md](../../../METRIC.md) and the root README now describe that aggregation; the imported block specification is preserved as a historical document. See [metric-source.json](metric-source.json).
45: 
46: ## Cleanup
47: 
48: Local cleanup removed the task-created publishing checkout and incomplete archive worktree only after checking the published GitHub commit and retained server archives. Available local disk increased by **3,624,886,272 bytes** at measurement time. See [local plan](cleanup-local.plan.json) and [receipt](cleanup-local.result.json).
49: 
50: The [server cleanup plan](cleanup-plan.json) was executed exactly: **154 derived prediction arrays and 24 aliases removed, no skips or errors**, reclaiming **54,534,419,712 logical bytes (50.789 GiB)**. The cache now contains 112 regular files and eight aliases, totaling 46,303,304,669 bytes (43.123 GiB). All 120 retained entries matched their original fingerprints. Checkpoints, experiment logs, datasets, foundational training metadata, current C/V1 controls, Transformer and prior M01/I1 inputs were preserved.
51: 
52: See the [execution receipt](cleanup-execution.json), [post-cleanup inventory](cleanup-after-inventory.json), and [complete immutable audit](cleanup-receipt/), including per-file SHA256 recorded before guarded deletion and regeneration mappings. All 220 source checkpoints and seven generator scripts were checked before execution. Rebuilding deleted predictions costs inference compute and is not guaranteed to recreate bit-identical bytes across hardware/software. The GitHub archive remains the historical pre-cleanup snapshot; this receipt explains which derived caches are no longer present on the live server.
53: 
54: No new model, competition submission or public score was produced. The next allocated research work should be the head-FT intervention; architecture and quant feature ideas first need their bounded feasibility checks.
```


## E14 — Independent checks

```json
{
  "environment": {
    "numpy": "2.3.5",
    "torch": "2.10.0+cpu"
  },
  "atom_decoder": {
    "atoms": [
      [
        -0.65687,
        0.0
      ],
      [
        0.65687,
        -1.97061
      ]
    ],
    "true_posterior": [
      0.5,
      0.5
    ],
    "weighted_CE_optimum": [
      0.19999999999999998,
      0.7999999999999999
    ],
    "metric_optimal_prediction": [
      0.0,
      -1.97061
    ],
    "current_decoder_prediction": [
      0.3941219999999999,
      -1.97061
    ],
    "untilted_decoder_prediction": [
      -2.533956079726158e-17,
      -1.9706100000000002
    ],
    "scope": "Decoder consistency, not a prediction of real-data score impact."
  },
  "linear_chunk_invariance": {
    "max_abs_full_vs_chunk24": 0.0,
    "max_abs_full_vs_chunk1": 2.9802322387695312e-08
  },
  "conv_chunk_invariance": {
    "max_abs_full_vs_chunk24": 0.012087330222129822,
    "max_abs_full_vs_chunk1": 0.022514864802360535
  },
  "dual_head_only_freeze": {
    "trainable_modules": [
      "event_gru",
      "head"
    ],
    "event_gru_trainable_parameters": 1116
  },
  "invalid_forecast_probabilities": [
    {
      "path": "E-316.result.forecast.P_gt_0.6641",
      "value": -0.04999999999999999
    }
  ]
}

```

Existing selected test run:

```text
.......................................                                  [100%]
39 passed in 1.35s

```
