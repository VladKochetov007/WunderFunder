# Readout transport size correction

R1's1MiB/file cap is below the real indented ancestry layout. A structural
lower-bound fixture from actual raw112_d0_s101101 run metadata, its required
producer.identity data/source files, config and1873prediction entries serializes
to1,198,465bytes at indent2, before other required fields. This fixture generated
no predictions and is not an actual grade. R1 native training/readout sources,
receipt history and publication state remain unchanged.

Three NEW files are byte copies of R1 with only these changes:

- `collect_l4_readouts_r2.py`:4MiB/file instead of1MiB. Total response remains8MiB;
  known source digests let subsequent requests receive remaining exact receipts.
- `l4_readout_mirror_r2.py`: imports the new finite size constant and queues the
  new collector source. Exact names, raw-byte hashes and immutable local files
  retain the original checks.
- `report_l4_wave_r2.py`: imports the new mirror. Same publication/state schema,
  same API helper/account verification, same source/config/grade gates.

ROOT stops only its exact R1 reporter, then launches the identical tested argv
with entrypoint `report_l4_wave_r2.py`. Keep the SAME state/readout/mirror files
and credential process environment. Never reset comment history or pending POSTs.
No existing reporter/source/controller file was edited by the critic.

Two focused size tests check exact bytes across several responses, original1MiB
refusal,4MiB/file refusal and8MiB total. The existing reporter tests cover hashes,
names, pending RPC/restart, partial results and uncertain POST deduplication.
