# Fold-general endpoint grading

- [grade_endpoint.py](grade_endpoint.py) verifies fixed checkpoint/run/fit ancestry and export hashes, exact evaluation row order, original labels, sequence indices and `need_prediction AND is_scored` against supplied parquet; it uses the canonical `global_wp_clip2_v1` accumulator and optional paired component bootstrap. These are exposed-fold conditional results, not CP30 replication inference.
- [test_grade_endpoint.py](test_grade_endpoint.py) contains synthetic rejection and metric fixtures; passing tests are not a completed real endpoint grade.

Root owns endpoint export and collection. The grader was implemented in a separate Luna/max session; independent review remains separate from authorship.
