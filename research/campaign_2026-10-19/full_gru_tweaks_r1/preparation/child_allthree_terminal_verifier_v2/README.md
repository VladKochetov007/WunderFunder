# AllThree terminal verifier V2 body repair

V1's independent exact-body review rejected the frozen worker because its parent preflight required `FG19_APPROVED_PATH_COMPAT_EXCEPTION=pathlib._local.PosixPath:exact-dictionary-path-only`, but the `systemd-run` support unit did not receive that variable explicitly. V2 copies the exact 706,653-byte V1 body and adds that one `--setenv` argv item immediately after `--setenv=CUDA_VISIBLE_DEVICES=`. The support unit, native AllThree identity, checker API, and other worker behavior remain byte-identical.

`V2_BINDING.json` pins the V1 body and rejection, root preparation, typed and raw native terminal evidence, and all fifteen V1 payload identities. The fifteen payload joins remain inherited from V1's static root payload check: the exact inverse-byte relation proves the payload-bearing body bytes did not change. This repair did not decode those payloads or read model, state, dataset, cache, or array contents.

The narrow local fixtures parse both worker files with `ast` without importing or executing either one. They verify the one-item argv insertion, compare full module ASTs after removing that item, and reject missing or duplicate insertions. Ruff format/check and these three focused tests passed.

This packet is local preparation only. It does not authorize staging, admission, or execution. The exact V2 body still requires independent source review and root's exact stdin/transport binding before any remote action. V1 and its rejected review remain unchanged.
