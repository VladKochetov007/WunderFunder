# Score exact-stop V4 conditional packet

V4 is the source-reviewed score-v3 stop namespace. Its preparation review is conditional on fixed ROOT acceptance/body/review/invocation pins and explicitly retains the limitation that local `capture_output=True` does not impose a pre-capture SSH stream cap.

- [V4 conditional source review](SCORE_STOP_SOURCE_REVIEW_V4.json)
- [Narrow namespace proof](SCORE_STOP_V4_NAMESPACE_PROOF.json)
- [V4 ROOT capture driver](ROOT_CAPTURE_SCORE_STOP_V4.py)
- [V3 template](STOP_SCORE_TEMPLATE_V3.py)
- [Actual bound stop source and transport reviews](../../records/first_cell_evaluation_v1/score_v4/exact_stop/README.md)

For the actual stop lifecycle, use the durable record and its conditional transport review V2. The earlier transport review V1 rejection is preserved there as a historical finding.
