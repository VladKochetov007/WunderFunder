# R10 terminal capture

[Raw V2 envelope](REMOTE_CAPTURE_RAW.json) carries nine small source/scalar/log payloads, extracted as REMOTE_* bytes after base64 length/SHA validation. [Validation](VALIDATION.json) preceded [exact stop](STOP_RECEIPT.json); REMOTE_STOP_RAW.json preserves the full pre/post observations. CAPTURE_STDOUT.raw is the empty failed V1 transport and CAPTURE_STDERR.raw its omitted-empty-property KeyError; CAPTURE_V2_* contains the corrected read-only capture. No data/model/geometry query was replayed. Terminal cgroup is explicitly empty, with no host-root fallback.
