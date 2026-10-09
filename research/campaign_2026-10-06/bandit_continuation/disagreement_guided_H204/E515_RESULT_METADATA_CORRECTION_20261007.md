# E-515 result metadata correction

The frozen `E515_RESULT.json` omits a `bandit_arm` field. Its registered
hypothesis and arm are unambiguous: H-204,
`arm:disagreement-guided-training`. The result JSON is preserved unchanged;
downstream experiment and tree records bind this result to that arm.

This is a metadata correction only. It does not change the exposed
disagreement/error-location statistics, E-515's limited claim, or the need for
ancestry-clean group-OOF predictions before any guided student training.
