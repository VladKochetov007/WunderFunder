# Independent package review

Reviewer: `team_source_critic`, native gpt-6-luna/max, separate from the builder.

Byte-level audit passes for ZIP SHA256 `4315eaa9bfe985fdaaca4258b3e8da3c4861264115f048bd8419b1cd6375f21a`, size 16,320,815 bytes. The reviewer streamed every ZIP entry through stdlib ZIP/CRC and SHA256 checks, verified exact sorted entry order, safe paths, no duplicates, root `solution.py`, intended member set, and matching sizes/hashes in the build manifest.

The selected weight entries were independently hashed directly inside the original `pair_ftt2` and E ZIPs, then compared with the authority, inventory, and delivered ZIP. The order is distilled seed6, FTT seed6, Z4, H390 seed3, each coefficient0.25. Callback source, precision file and all three native libraries exactly match their entries in the original `xteam_h4` ZIP. The callback passes `budget_s=None` and loads a fixed member list.

The build manifest omits the recipe hash. This release therefore records it separately: recipe SHA256 `2622400b0a3f8f1e337afc1a9f4b0999de1c208ab66367687699045b034f7b34`; authority SHA256 `c994e70336bcf09d4582da671485e6e517392c99a7c4ee431f49cd502e99ea2d`. Both match the packaged membership exactly. Inventory SHA256 is `997b9eaef64da0d8f4d087ab9fcb7acbefbfb5382d394d0b218a36e3d7497a15`.

The reviewer performed no inference, timing or ABI execution. Root owns the separate actual-ZIP ABI, isolated Python3.11 import and repeated streaming checks. Package integrity does not establish a quality gain or public score.

The reviewer also identified the 5.6GB `data.read_bytes()` allocation in the older `check_submission_python311.py`; that checker was not used. The chosen callback-panel checker streams file hashes. An initial claim that its warmups contained NaN was withdrawn after checking the actual source: they are zero-filled, so repeated array equality is valid.
