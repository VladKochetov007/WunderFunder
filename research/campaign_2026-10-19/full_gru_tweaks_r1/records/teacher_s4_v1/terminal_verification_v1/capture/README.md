# S4 verifier terminal metadata capture

[Raw response](STDOUT.raw.json), [transport](TRANSPORT.json), [capture script](REMOTE.py), [reuse binding](REUSE.json), [stderr](STDERR.raw) and [validation](CAPTURE_VALIDATION.json) preserve one bounded readonly capture after the exact verifier exited0. Five compact payloads were decoded only after raw persistence and independently length/SHA checked: [verification](VERIFICATION.json), [verifier admission](VERIFIER_ADMISSION.json), [verifier launch](VERIFIER_LAUNCH.json), [native exit evidence](TERMINAL_EVIDENCE.json), [empty unit log](unit.log). No tensor, model/state/data payload was copied toMac.

The verifier's `updates`/nested `source` layout differs from a first local display mapping's assumed flat fields; that local KeyError was corrected from durable existing bytes. No verification/capture/SSH was repeated for the mapping. Successful native and support exit/accounting fields and parent events remain original. The root [acceptance](../../TERMINAL_ACCEPTANCE.json) precedes the [exact stop](../../terminal_exact_stop/README.md).

[Empty capture stderr](STDERR.raw) is retained separately from the empty native support unit log.
