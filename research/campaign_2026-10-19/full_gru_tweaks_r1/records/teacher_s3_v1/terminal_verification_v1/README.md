# S3 terminal verifier V1 — preserved failure

Verifier V1 failed as a consumer at restricted checkpoint loading: `weights_only=True` rejected the canonical `torch.torch_version.TorchVersion` metadata object. The failure is not evidence of a training failure; native S3 had already exited successfully and its checkpoint was left unchanged.

- [Failure record](FAILURE.json), [transport](TRANSPORT.json), [submitted checker](REMOTE_SUBMISSION.py), and captured [stdout](STDOUT.raw.json)/[stderr](STDERR.raw).
- The [V1 capture details](capture/README.md) preserve the failed unit's logs and receipts.
- The narrowly scoped compatibility correction passed in [V2](../terminal_verification_v2/README.md); V1 remains preserved as the failed first verification attempt.
