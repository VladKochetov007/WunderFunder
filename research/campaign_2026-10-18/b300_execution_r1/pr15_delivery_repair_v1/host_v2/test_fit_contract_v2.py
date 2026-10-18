"""Ensure the repaired fit checker pins itself and its payload comparator."""

from __future__ import annotations

import fit_validation


def test_v2_fit_contract_binds_validator_and_parent_profile():
    contract = fit_validation._check_contract()

    assert contract["schema"] == "tabdata-pr15-fit-contract.v2"
    assert contract["profile_packet_sha256"] == (
        "c18f1877b8579c03d2a0905534f5ecbd13ed56a54306ba66d63b47d892b66510"
    )
    assert contract["parent_fit_contract_sha256"] == (
        "05aa087af1fdd68ac97146e6f56750157d4a33ebd4fce2a5ec2f5eb30cc6bcb8"
    )
    assert set(contract["files"]) == {"fit_validation.py", "checkpoint_identity.py"}
