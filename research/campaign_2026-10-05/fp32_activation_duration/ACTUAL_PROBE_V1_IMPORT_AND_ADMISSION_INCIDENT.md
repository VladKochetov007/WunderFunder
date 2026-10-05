# E489 actual probe v1 import failure and late admission

Actual attempt `e489-fp32-duration-real-gpu-probe-v1` failed at module import: `ModuleNotFoundError: No module named fit_activation_placement`. The probe's explicit sys.path omitted activation_placement and imported its module before the actor could add that directory. Synthetic actor tests/preflight had populated that path, so they did not test this direct entry point. No CUDA probe result, full training trajectory or evaluation score exists. Original sealed source, review, attempt and traceback stay unchanged. A versioned source-only repair and direct CLI regression from/tmp are prepared for the next research window; no actual GPU retry here.

Root also missed the new-admission cutoff: controller reservation was07:00:32.527333UTC and worker began07:00:36.236196UTC, beyond07:00 by32.527seconds. This was a controller deadline failure, not an exception or a claim that admission occurred before cutoff. It exited at07:00:37.858218UTC. No full fits followed; further scientific admissions are closed. Future launch must recheck the UTC deadline immediately before reservation, after source hashing, rather than relying on an earlier clock observation.

The previously completed12E488trajectories will still be graded and documented as drain work. E489 remains six prepared, unadmitted full fits.
