# Private import-boundary proof V2 scope

`PRIVATE_IMPORT_BOUNDARY_PROOF_V2.json` is 12,712 bytes, SHA-256 `d590fccb9c001791cdd3656a423e424a8d0042cf3a33e2415c381b5a1d306a11`, schema `fg19.child-allthree-terminal-import-boundary-proof.v2`, status `PASS_NARROW_ALLTHREE_PRIVATE_IMPORT_BOUNDARY_V2`. Its harness is 23,177 bytes, SHA-256 `c9da47db67885e569cf18a94eea7266d0733bbbbea8cbbdb20bd8bda6418d512`.

The fresh child process copied and checked the exact seven pinned code files before and after import, rejected preloaded private modules, confirmed private bytecode was absent before and after import, and verified each module origin against the private temporary copy. It called only `assert_helper_closure`; `verify_terminal`, CLI/main, and fit/evaluation functions were not called. The proof reports zero protected worker/data/model opens, zero guarded CUDA attempts, zero WNN source modules, and no CUDA initialization.

This is a narrow import-boundary result. It does not execute the full checker, read any model/state/dataset/cache payload, prove the remote Torch runtime, or authorize a support unit. The synthetic schema fixture separately reads only the pinned 3,140-byte NLL stop metadata receipt and checks its identity and strict hyphenated schema.
