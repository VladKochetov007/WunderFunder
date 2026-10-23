# EMA terminal verifier V2: preserved helper-binding failure

One reviewed V2 support was admitted after fresh native/source/resource gates. It passed the original dictionary stat schema check, then exited1 because `PARENT_ACCEPTANCE_SHA` was referenced from the wrong helper module. The traceback precedes parent/dictionary/model/state payload loading. Native EMA remains exited0; full integrity acceptance is pending. Failed V1 and V2 are preserved and excluded from retry/restart/stop.

- [Reviewed body](REMOTE_SUBMISSION.py)
- [Root admission manifest](ROOT_ADMISSION_MANIFEST.json)
- [Launch acceptance](LAUNCH_ACCEPTANCE.json)
- [Raw response](STDOUT.raw.json)
- [Raw stderr](STDERR.raw.txt)
- [Transport](TRANSPORT.json)
- [Admission](VERIFIER_ADMISSION.json)
- [Launch](VERIFIER_LAUNCH.json)
- [Failure capture](capture/README.md)
