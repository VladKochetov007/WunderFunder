# E-508 v5 independent packet review

**Verdict: PASS.** The review found no blocking issue. Scope is packet correctness only; it does not validate a profile result, scientific fit, score, or public gain.

- At pinned trainer commit `69b2331825e193a070015ca4182872724595209a`, `scripts/train_rnn.py` accepts `--device cuda` or `cpu`. V5 uses the supported `cuda` selector; v4 used unsupported `cuda:0`. The v4 and v5 profile/fit argv differ only at this token.
- The worker AST matches v4 after normalizing only the packet ID and device token. The internal `torch.device("cuda:0")` remains unchanged.
- Embedded worker SHA-256 is `4929ff3ce874942de5b4f92a9b59013e72468e275eff723bc722b4a50e0b92a2`; embedded source archive SHA-256 is `78f9e5c3d001b8a0254ed535846f07b3b9a906adba2a270f7ddc1ab6606081fc`. Both match the build receipt. All 47 source archive members match their recorded hashes; pinned trainer hash is `8a29459e998f7f5d6a03b7a4b87c6f4b569002e65b8c87127c37138360909cbf`.
- Metadata is private, internet-disabled, GPU-enabled, and TPU-disabled. `PROFILE_ONLY = True`; the worker exits with `fit_admitted=false`, so this packet cannot start either teacher fit.
- V4/v5 frozen recipe, fold, input map, exports, profile gate, guardrails, and immutable input identities match. Fold membership remains 10,535 train groups and 1,405 valid groups, excluding 468 evaluation groups and corrupt train group 528.
- The profile gate remains 3,600 seconds worker time, 4,200 seconds provider watch, and a 4.3 GB allocator cap. Focused command, worker-equivalence, and materialized-packet tests passed (3/3 in the independent review; the full focused packet suite also passed locally).
- V4 terminal evidence is `SystemExit(2)`, no fit, no OOM, and zero CUDA allocation. It is an execution failure, not scientific evidence.

Reviewer: independent native Luna max packet critic (`/root/e508_v5_packet_critic`). The reviewer made no code or packet changes. Review was completed before v5 admission.
