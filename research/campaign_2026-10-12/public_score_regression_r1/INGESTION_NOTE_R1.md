# Portal roster ingestion, R1

`raw_portal_roster.tsv` is a literal transcription of the 42 rows supplied by
the user on 2026-10-12. The portal reported 45 total entries; the other three
are not visible in the supplied capture and remain unknown. The displayed time
strings are retained verbatim; no timezone is inferred.

The source counted 38 `SCORE`, 3 `TIMEOUT`, and 1 `CHECK_ERROR`. Blank score and
runtime fields are missing outcomes. The three timeout durations are retained
as censored/failed-serving times, not completed prediction times. The
`20o154uy` row is retained as a scored result with its reported 61:22 duration.
The `10tq248b` filename is `submission.zip`; a prior source note identifies the
package alias `c20261004_combo_s3_i16`. The ZIP alias does not replace the
portal filename or submission ID.

All portal values are user-reported. This file is outcome data, not a verified
portal API export. No outcome field (score, status, duration, memory) may be a
predictor input for that same upload.
