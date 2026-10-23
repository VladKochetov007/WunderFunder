# Dictionary-only child admission template, renderer V2

This version preserves the frozen dictionary-only recipe, admission template, original README, and V1 renderer/check receipt. V2 supersedes the V1 renderer after its reviewer found that the admission and launch predecessor links share one line anchor, so a single-occurrence replacement could not render the actual template.

The V2 renderer patches the admission and launch predecessor links through their distinct surrounding contexts. It also tightens the admission-only scratch gate to require the exact physical `B/research_tmp` path, ext4, and device 64512 matching the project root, while retaining the 25 GiB free-space minimum. It records that identity in the admission metadata and leaves data/cache path alias rules unchanged. Its local check exercises the predecessor and scratch rewrites with small synthetic source fragments only; it reads no EMA acceptance or exact-stop bytes and does not create an executable body. It also checks that the base EMA body still contains each distinct frozen anchor exactly once.

The run remains `fg19_child_dictionary_f0_s1001_fit_v1`, unit `computer-vision-homework-fg19-child-dictionary-fit-v1.service`, sparse mode, WMSE, seed 1001, B16/C2000, and six fixed epochs. In the trainer argv, only the run name and representation value (`ema` to `sparse`) change. Initialization remains the accepted parent checkpoint at epoch 3 with inherited normalization. The exact shared dictionary artifact must pass its fresh path, size, and digest gate before trainer invocation; missing or mismatched bytes fail closed before the trainer can fit a replacement.

The actual EMA terminal acceptance and separate exact-stop receipt are still unbound. The native exit capture alone cannot satisfy the predecessor gate. Do not render `REMOTE_SUBMISSION.py` until root provides and validates both actual receipts for the exact EMA invocation. The renderer then validates the six-epoch counters, all terminal file metadata pins, selected model/training-state tensor identity proof, finite model fingerprint, source, optimizer, normalization, RNG, sample order, ancestry, cache, and successful exact stop before exclusive body creation. Serialized `model.pt` and `model_ep5.pt` digests are checked independently because their byte hashes may differ even when the verifier establishes tensor identity.

Run the V2 local-only check with:

```sh
./.venv/bin/ruff format research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_dictionary_fit_v1/bind_dictionary_fit_v2.py
./.venv/bin/ruff check research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_dictionary_fit_v1/bind_dictionary_fit_v2.py
./.venv/bin/python research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_dictionary_fit_v1/bind_dictionary_fit_v2.py --check
```

The check is local and metadata/code-pin only. It does not stage, query, admit, fit, evaluate, or stop anything.


## Binding update — 2026-10-20 18:04 UTC

Root supplied and the renderer validated the actual EMA terminal acceptance `records/child_ema_fit_v1/TERMINAL_ACCEPTANCE.json` (13,016 bytes; SHA-256 `0aea6295b64670bb02ac008fdfece6ee430b8f68b84904a14c39dd816c0161a8`) and separate exact stop `records/child_ema_fit_v1/exact_stop/EXACT_STOP.json` (3,093 bytes; SHA-256 `548fe9876947345d4ac443601e478341bdfe165d5670d7397d50b401b93cc2bf`). The exact stop binds the acceptance digest and confirms the successful EMA invocation is stopped.

The V2 renderer produced `preparation/child_dictionary_fit_v1/REMOTE_SUBMISSION.py` locally (196,111 bytes; SHA-256 `9f545489dd80c13831b1a02c75dddf4059f3cbfb84311b2400db4830b4fdd77e`). Its body embeds those exact receipt bytes and preserves the accepted parent epoch-3 checkpoint as initialization. This is a rendered admission body awaiting independent final-byte review; it has not been staged or admitted, and no dictionary worker action was performed.


The post-render local integrity check completed at 2026-10-20T18:05:31Z. `DICTIONARY_RENDERED_BODY_LOCAL_CHECKS_V2.json` is 1990 bytes (SHA-256 `c238520fb138dda633befdb7980d0b637d96a3962aeeb51befeea507762e7fba`); it independently parses the body without importing or executing it, checks the exact embedded EMA receipt bytes and stop linkage, and confirms the 92-element trainer argv differs from the EMA wrapper only at run-name and representation-mode positions. The body bytes above remain unchanged. This is local integrity validation only; independent body review and root-owned admission remain pending.


### Supersession note

The earlier V2 paragraph describing the EMA receipts as unbound, and the pre-render `--check` command above, are historical preparation guidance and were superseded by the binding update and post-render check above. The actual receipts are bound in the body and manifest. Because `REMOTE_SUBMISSION.py` now exists, the pre-render `--check` must not be rerun; the body remains local, unstaged, and awaiting independent source review.
