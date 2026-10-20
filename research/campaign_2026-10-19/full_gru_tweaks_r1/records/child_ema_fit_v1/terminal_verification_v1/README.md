# EMA terminal verifier V1: preserved engineering failure

The reviewed body was admitted once. Native EMA remained `active/exited` with exit 0. The CPU support exited 1 on dictionary metadata schema comparison: the original receipt contains no `mtime_ns`, whereas the V1 checker expected it. This is not a native fit or quality failure. The failed unit and exact artifacts remain preserved; no retry, restart or stop is authorized for V1.

- [Root one-admission manifest](ROOT_ADMISSION_MANIFEST.json)
- [Reviewed exact body](REMOTE_SUBMISSION.py)
- [Actual launch acceptance](LAUNCH_ACCEPTANCE.json)
- [Raw SSH admission output](STDOUT.raw.json)
- [Transport](TRANSPORT.json)
- [Actual admission](VERIFIER_ADMISSION.json)
- [Actual launch](VERIFIER_LAUNCH.json)
- [Captured result](capture/README.md)
