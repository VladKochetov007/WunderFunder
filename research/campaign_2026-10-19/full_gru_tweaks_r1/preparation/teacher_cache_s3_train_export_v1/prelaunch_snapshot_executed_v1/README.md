# Prelaunch packet snapshot

This directory preserves the packet bytes immediately before the S3 train-export admission. The body, template, and preparation file are byte-for-byte copies of the canonical packet files in the parent directory.

The `REMOTE_SUBMISSION.py` bytes here (SHA-256 `976cb7efe836b428e9944e68ea9bad5d053e27f71dd3e55cd3536dc0c65d38c6`) were launched once at 2026-10-19 22:12:29 UTC as `computer-vision-homework-fg19-teacher-s3-train-export-v1.service`, invocation `48732136d70548b2903f3426bfedf656`. Therefore this is an executed prelaunch snapshot, not an unlaunched candidate.

The separately recorded path-check packet is under the parent directory's `PATH_CHECK_*` files. No later admission guard variant is represented here. The original snapshot filenames and file contents are preserved unchanged.
