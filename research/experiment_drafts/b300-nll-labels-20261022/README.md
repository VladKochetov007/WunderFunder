# Cold-label mean/logvariance NLL screen

This packet specifies a small, matched objective screen. It does not claim to reproduce the warm MainT3 NLL run, and it contains no B300 fit or measured runtime.

The code adds an explicit `--mean-variance-target-source labels` path to the canonical trainer. The default remains the historical MainT3 blend. Both study arms add the same two zero-initialized log-variance outputs; WMSE ignores them, while NLL learns from observed labels only.

Read [the experiment proposal](PR.md) and [the run handoff](COMMANDS.md) together. The intended four fits are F0/seed1201 and F2/seed1202, each with a WMSE/NLL pair. Admission remains blocked on source-qualified split-manifest paths and hashes, data and package/environment pins, a measured B300 profile, and the external grader invocation. No fit should start from the placeholder commands until those values are bound in the run receipt.

The feature lives in `src/wnn/training/{cli_targets,mean_variance,losses}.py`, with focused tests at `tests/test_mean_variance_labels.py`. Source base is `68ff682977b822f38351aac0f621146f5f13b440`; the experiment branch's final code SHA is recorded after the local commit.
