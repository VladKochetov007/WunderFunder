# B300 QK RMS receiving record

**State:** four fresh two-rank fits were admitted and launched. The last
transferred training snapshot is stale; current process and GPU state are
unverified. No completed fit, validation score, or quality claim is recorded.

This is an execution addendum to the submitted PR #75 draft. The frozen source
and recipes remain unchanged. The original PR body in `../PR.md` is retained as
the submitted snapshot; this file records the later receiving amendment.

## Frozen identities and finite clock

| Item | Identity |
|---|---|
| Native root | `/data/epanov_s/tabdata-qknorm-b300-20261111-r1` |
| Source commit | `8626000a4b965763a70011a0291a891aa6535d7e` |
| Source capsule SHA256 | `931a91046efb0f84e68184862e80076bb2751522bc6264fffa06e3cab36931fd` |
| Source manifest SHA256 | `cb154b75d67fa63740be93264386644bcaab8f7f13952cb7e73935b49b612052` |
| Authority R2 SHA256 | `b29dd5a08bdd75371808ce99c6b95730a864581b31b720cd6d52bae2b9f64837` |
| Fit admission SHA256 | `849b396a54c2e4ffbc5eeff572ddbcbb1921bdd019999fa951062ae28b815dc6` |
| Primary split SHA256 | `0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c` |
| Split members | 10,325 train; 1,873 validation; 3,528,273 scored rows |
| Fixed endpoint | E4, 3,280 updates per fit; world size 2 |
| Fit / wave deadlines | 2026-11-11 05:10:28 UTC / 06:10:28 UTC (08:10 / 09:10 MSK) |

The preserved first-profile/GPU epoch is `1794359428.5272088`. The five-hour
wave clock includes four hours for fitting and one hour for endpoint collection.
The actual profiles recorded Python 3.12.14, PyTorch 2.14.0+cu130, CUDA runtime
13.0, and NVIDIA B300 SXM6 AC devices. Forward compute used the frozen
`bf16_bf16x9` mode with FP32 parameters, gradients, optimizer state, and loss.

## Admitted cells

| Cell | Arm / seed | B300 pair | Normalized config SHA256 |
|---|---|---:|---|
| `plain_s203301` | plain / 203301 | 0, 1 | `00106b3fec754993536db2ccf69d889fe4d344a5a7d9ba4faeb01b7c0b757c37` |
| `rms_s203301` | RMS / 203301 | 2, 3 | `b0188ce1b62ce01345c3c4ff64cf728614562c5497943ce3f81b78d9bc795f17` |
| `plain_s203302` | plain / 203302 | 4, 5 | `3aacdfccd8cc41a1cbb6883f26fa841108c4e6226dbfad6cbae66f9559a99a63` |
| `rms_s203302` | RMS / 203302 | 6, 7 | `0d478b018075b328a748e4cf06fe4a5a164b2e611d7402e2719907e791e37fff` |

ROOT amended physical placement so each seed's plain and RMS cells use
separately qualified, identical-class B300 pairs. Exact software, data,
configuration apart from the declared QK norm and seed, world size, precision,
and fixed-E4 endpoint remain bound by the authority/admission receipts. This is
not same-device-pair replay; the comparison receipt allows only the explicitly
admitted device UUID difference while retaining each recorded runtime.

## Profile and source-native receiver gates

All eight engineering profiles passed before fit admission: each cell completed
the 50-update production-warmup profile and the 50-update warmup-zero peak-LR
diagnostic. Full model/optimizer/scheduler/RNG state reload equality passed on
both ranks. ROOT's profile review records a maximum reserved allocation of
125,101,408,256 bytes (116.50 GiB) and maximum per-rank host RSS of 6,927,257,600
bytes (6.45 GiB). These are profile observations, not fit-peak measurements.

The first receiver fixture/summary adapter required correction before release.
The frozen R2 receiver archive and its actual-producer fixture supersede that
attempt; ROOT's independent R2 review records the producer identity, exact
evaluation coverage, UUID-only runtime amendment, finite clock, and fit-owner
exit/lock-release gates. R2 did not alter the four fit configurations or source
capsule. The pinned readout receiver was transferred before the post-fit waiter
was started.

## Launch and last verified progress

The native R2 profile controller was launched from the approved capsule at
01:31:35 UTC ([RPC 065](native-receipts/065-qk-fourpair-profiles-launch-r1.json));
the terminal profile receipt is [RPC 067](native-receipts/067-qk-profiles-terminal-check-r1.json).
ROOT approved the exact four-cell fit admission ([RPC 069](native-receipts/069-qk-ROOT-four-fit-admission-r1.json)).
The four-fit controller was launched at 01:42:44 UTC with admission SHA above
([RPC 070](native-receipts/070-qk-four-full-fits-launch-r1.json)); its owner PID
and birth identity were recorded as `2425084 / 177125262`. [RPC 075](native-receipts/075-qk-four-fit-first-updates-r1.json)
captured finite training progress on every cell. A corrected owned resource
observer was attached under R2 ([RPC 076](native-receipts/076-qk-correct-resource-observer-r2.json));
the observer is monitoring evidence and does not create or change fits.

The last mirror state applied at **2026-11-11 02:10:51.697 UTC** (05:10 MSK).
Its progress cursors recorded:

| Cell | Last transferred update | Fixed endpoint |
|---|---:|---:|
| `plain_s203301` | 883 | 3,280 |
| `plain_s203302` | 890 | 3,280 |
| `rms_s203301` | 912 | 3,280 |
| `rms_s203302` | 919 | 3,280 |

ROOT then reported that both SSH streams had reset after the OpenConnect/VPN
connection disappeared; one bounded reconnect attempt per stream timed out at
the banner. There is no newer training, process, or resource proof in this
packet. Treat the jobs as **last observed progressing at 02:10:51 UTC**, with
current liveness unknown. The mirror bridge state is included as a transport
cursor receipt; its cursors point to the native log tails but do not contain the
full journals.

## Standalone full-validation readout

The bounded, detached post-fit waiter was launched at 02:03:31 UTC
([RPC 078](native-receipts/078-qk-bounded-postfit-readout-waiter-r1.json));
PID/birth `2431727 / 177249993`). It is tied to the original 06:10:28 UTC wave
deadline and admits **zero new fits**. It waits for all four exact fit receipts;
partial/failed endpoints do not qualify for readout, and retries are disabled.
The pinned R2 readout package transfer is recorded in [RPC 077](native-receipts/077-qk-canonical-readout-and-waiter-r2-source.json).

On full fit completion, the canonical source's standalone `predict_sequence.py`
exports all 1,873 `evaluation` members for each cell to
`runs/{cell}/evaluation/`; canonical `score_sequence.py` writes
`runs/{cell}/fit/score.json`. ROOT's R2 receiver expects complete manifests,
3,528,273 scored rows per cell, matching model/source/config/data identities,
then writes `control/qknorm-receiver-r1/readout-complete.json`. Only when all
four endpoint grades pass does the fixed pair comparison write
`comparison-fourpair.json`. No validation-selected checkpoint, blend, refit, or
extra endpoint is added. The standalone full-validation component set is
exposed development data and cannot establish public-LB improvement.

## Included byte-bound evidence

| Artifact | SHA256 |
|---|---|
| [AUTHORITY_R2.json](AUTHORITY_R2.json) | `b29dd5a08bdd75371808ce99c6b95730a864581b31b720cd6d52bae2b9f64837` |
| [SOURCE_R1.json](SOURCE_R1.json) | `c5573ed7974e34694940c21187533b6d49c75400786c087bf5460f16c9a0f1d3` |
| [ROOT-fit-admission-r1.json](ROOT-fit-admission-r1.json) | `849b396a54c2e4ffbc5eeff572ddbcbb1921bdd019999fa951062ae28b815dc6` |
| [ROOT-profile-review-r1.json](ROOT-profile-review-r1.json) | `e989785943d63fa12db55029170125cd94d40f8187d877111f6179aecb8c9758` |
| [RECEIVER_R2.json](RECEIVER_R2.json) | `e0e61ff6be2569a7672e062a0b03d80ea202912fa485bfb4640792c50a5eeef8` |
| [ROOT_READOUT_RECEIVER_R2.json](ROOT_READOUT_RECEIVER_R2.json) | `7b51d71f0c077ec21986a08b6589ddc51079df5b82f1c8a574e28698e6309ee1` |
| [native-profile-evidence-r1-manifest.json](native-profile-evidence-r1-manifest.json) | `93f5e01dced35adfbd4d6348d69f75d6b4508f709781f09cdd64cd18f9088b1b` |
| [clock-r2.json](clock-r2.json) | `b5523b07db6398d39108f294355e37f2c67b1742a968445e3ceadf85e736faa9` |
| [profile receipts archive](archives/native-profile-evidence-r1.tar.gz) | `21e1aafd73d2068228ff3dd0a2dd2e5d7e5de500a2fd91e54f6be6177e8ab83a` |
| [readout receiver archive](archives/readout-receiver-r2-root.tar.gz) | `1efcc17cc06fcab6e2e59bef92a89e3471fb6922defd3486eff42f371cbe598c` |
| [last mirror bridge state](last-bridge-state-r1.json) | `05682a63551bc79870260f567f12495df553f701018ed7e07dee7d41549b97c1` |

The `native-receipts/` directory preserves RPC evidence for profile launch and
completion, ROOT fit admission, four-fit launch, first updates, observer
attachment, readout receiver transfer, and waiter launch. `SOURCE_R1.json`
identifies the immutable source capsule; the working-tree documentation commit
is not part of the scientific source identity.

No result is complete until the native fit/readout terminal receipts and their
model/config/source/data hashes are collected and checked. Keep every partial
checkpoint and journal if a transport timeout or deadline leaves any cell
incomplete; transport failure is not a negative QK result.
