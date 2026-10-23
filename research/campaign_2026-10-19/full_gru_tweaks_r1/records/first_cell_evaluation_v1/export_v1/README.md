**Independent audit:** [Actual preflight failure and exact absence metadata review](../failedexport_v1/README.md) passes without B queries or payload replay.

---

# First export preflight — 21 October08:39UTC

One actual source-reviewed admission-body V2 was sent at ROOT08:39:54.728693–08:39:55.373668UTC. [Transport](TRANSPORT.json)729B/1abf3a11 records exact stdin243608B/a15b2cec and independent source review3289B/314a374d, SSH1/no timeout, [stdout](STDOUT.raw.json)0B and [stderr](STDERR.raw.txt)769B/6aab5654. Both raw byte streams and transport were fsynced before any parse. The traceback fails `check_support_helpers` on the incorrect nested `fg19_child_allthree_verify_v2/preparation/parent_fit_terminal_check_v3.py` path, before private-directory staging and systemd submission.

One [read-only exact outcome diagnostic](preflight_outcome_v1/README.md) proves the new export unit is not-found/inactive/dead/MainPID0/emptyinv and its private output/evidence directory does not exist. Three saved helper files are directly under successful CPU2's evidence root, with stat identities matching its original admission. No old fit/support unit or model/state/array payload was queried/read. This attempt created zero GPU/inference/support units and zero fits or quality results.

Next is a newly source-reviewed versioned packet: two binding helper paths lose the false `preparation/` prefix, exporter/scorer binding byte/SHA constants update, operational functions remain unchanged. Preserve this exact failed packet and raw transport; do not resend it. The same new worker unit/directory may be used only because their absence was actually observed, after fresh serial gates.
