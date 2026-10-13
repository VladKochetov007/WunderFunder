# QAT package sibling import repair

The submission template imported `qat_archive` before making its own directory available on `sys.path`. The existing offline benchmark/import consumers use `spec.loader.exec_module`, which does not establish the package directory automatically. Root static inspection found this before final ZIP dispatch.

The template now derives `HERE` and inserts it before importing its shipped sibling. This changes only import resolution. Actor qat_python independently reviewed the three-line change and returned GO for bounded actual-ZIP verification.

Validation: 14 generic tests passed with three optional fixtures skipped, then all17 passed in3.41s with the actual final S4 learned-W4 archive, exact native library, original xteam weights and portable native manifest. Root ran the fixture under1CPU/1GB/180s; no GPU fit or provider action. Template SHA256: `2c3fbba2b1c805480fc1c321242f3be74572f2a9d1320aaf3171ec7aaf4b62b2`. Final ZIP callbacks and benchmark remain separate checks.
