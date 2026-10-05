# W384 v6 immediate bootstrap failure diagnosis

The v6 provider run stopped at 2.08 seconds in the reviewed bootstrap's `validate_external_mounts`, raising `ValueError: External input mapping overlaps an archive member`. The failing check is the exact-key predicate `if logical in archive_members` in `kernel.py` (provider notebook traceback: `/kaggle/src/script.py`, line 52).

The uploaded v6 artifact identities were independently read from the local immutable bundle used for the push:

- `bundle.wnn`: `db08107d0df5ee9194959bbba4862385189d1fc0c91324ecd28ed12e8bebdd6e`
- `bundle-manifest.json`: `f89a77743159357ba735bbccf6385f73e79537715331bb578e2b833a2b6b6c40`
- kernel bootstrap: `c60dd550411bc9efbe858849b1f12a5136867c5599974075c1f3fef7297b496e`

Recounting actual regular-file tar members gives 265. The manifest has 287 `files` inventory entries, including all 26 declared external input paths. All 26 overlap the inventory set, beginning with `artifacts/runs/c20260929_prop_f1/model.pt`; none of the 26 overlaps the actual tar-member set. These paths are declared inputs whose bytes are supplied by the external dataset and intentionally excluded from the bundle archive.

The cause is that `_kernel_work` correctly extracts members while accumulating the actual archive members in `seen`, but then discards that set for the overlap check and constructs `archive_members = {item["path"] for item in manifest["files"]}`. The manifest inventory is not the archive-member set: it includes the externally mounted files. Thus the first external path is falsely rejected despite being absent from the archive. This is not an actual archive collision or a hash mismatch.

Narrow correction proposal: pass the already accumulated `seen` set into `validate_external_mounts` (or equivalently derive the set from actual tar entries), retaining the existing `logical in archive_members` rejection. This preserves rejection of real extracted-path collisions while allowing declared external inputs that are only present in the full inventory. Do not weaken path safety, manifest binding, mount descriptor equality, or target-existence checks. The current v6 artifact remains unchanged; this note does not authorize a rebuild, repush, or retry.
