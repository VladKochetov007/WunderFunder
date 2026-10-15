# Initializer-family F0 recipe files

This folder holds the frozen source-bound recipe for a small initializer-family comparison on the repeatedly exposed F0 panel. It is static preparation only: the source/metadata preflight, resource profile, fits, prediction exports, and grades have not run. No profile or fit is currently admitted.

- [CONTRACT_R1.json](CONTRACT_R1.json) pins the CP5 parent, clean-T3 cache and teacher ancestry, propensity weights, complete-component subset, objective, three arms, outputs, and resource ceilings. SHA-256: `97af50e9c815cc86de954b86ca39ee641f773758ee5e691a6a2f95ea698291a7`.
- [SOURCE_BINDING_R1.json](SOURCE_BINDING_R1.json) pins the exact trainer, initializer implementation, this recipe's scripts and manifests, and official grader sources. SHA-256: `4e7f2bfa5c6525fadf7764dd4ebbc915eef5398e2f87e82a312ff54dc2a5dac3`.
- [FIT_SUBSET_R1.json](FIT_SUBSET_R1.json) selects 28 whole fit-only components (104 complete sequences: 86 train and 18 valid) with no labels used in selection. SHA-256: `4559f2480de60ea22bc59ae81645d25ed02eeceba7e2fc911862cf2d6fa01dd`.
- [prepare_initializer_family_f0_r1.py](../../../../scripts/prepare_initializer_family_f0_r1.py) is the deterministic metadata-only subset builder.
- [run_initializer_family_f0_r1.py](../../../../scripts/run_initializer_family_f0_r1.py) implements the bounded `preflight`, one-update-per-arm `profile`, and fixed two-epoch `fit-export` stages. Its source SHA-256 is `4f45753ccf80efc462dfec5aa4bf25f9ec41b58a3768da6435c8f16086c9203d`. The subset builder SHA-256 is `5fe42b75ba0a580bd4af6085d4ce940f7bb580817241eb764a820b1b9fd58ffd`.

The recipe's status is frozen but not executed. In particular, Ruff format/check passed during preparation, while the runner's static `--phase preflight` has not been run; these are different checks. All later phases require a resumed research scope, a fresh resource admission, and a passing source/input preflight.
