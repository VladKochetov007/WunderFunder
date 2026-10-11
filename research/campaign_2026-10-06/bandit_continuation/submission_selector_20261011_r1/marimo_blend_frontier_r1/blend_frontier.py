import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import pathlib
    import marimo as mo
    root = pathlib.Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
    selector = root / "research/campaign_2026-10-06/bandit_continuation/submission_selector_20261011_r1"
    metrics = json.loads((selector / "generated_r6/F1_SELECTOR_METRICS_R1.json").read_text())
    return json, metrics, mo, root, selector


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Complete blend quality and serving frontier
    Our selection target is the complete deterministic endpoint. More members earn inclusion through conditional official-metric value. Weight storage, runtime precision, seed diversity and family quality are separate variables.

    **Evidence limits:** Fold 1 is exposed; final full-data predictions measure serving fidelity, not cross-validation. A precision loss measured on one pair cannot decide a different, larger blend. No public-score arithmetic is used.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    family_filter = mo.ui.dropdown({"All fixed mixtures": "all", "Seed averaging": "seed", "Student and teacher": "mixed"}, value="All fixed mixtures", label="Endpoint family")
    family_filter
    return (family_filter,)


@app.cell(hide_code=True)
def _(family_filter, metrics, mo):
    blend_rows = [{"endpoint": _item["candidate"], "official_wp": _item["full_fold"]["wp"], "target0": _item["full_fold"]["t0"], "target1": _item["full_fold"]["t1"], "evidence": "exposed F1, FP32 predictions"} for _item in metrics["fixed_mixtures_full_fold"] if family_filter.value == "all" or (family_filter.value == "seed" and _item["candidate"].startswith("h384")) or (family_filter.value == "mixed" and _item["candidate"].startswith("h256"))]
    mo.ui.table(blend_rows, selection=None, pagination=False)
    return


@app.cell(hide_code=True)
def _(mo):
    serving_rows = [{"candidate": "Teacher seeds 4/5", "transport": "float32 NPZ", "runtime": "Q8VNNI", "ZIP_MB": 17.272496, "host_us_per_row": 101.41, "members": 2, "heldout_status": "exact F1 recipe counterparts; exposed"}, {"candidate": "Student + teacher seed 5", "transport": "float32 NPZ", "runtime": "Q8VNNI", "ZIP_MB": 12.914019, "host_us_per_row": 74.81, "members": 2, "heldout_status": "student parent/since differs; provisional"}, {"candidate": "Student seeds 3/4", "transport": "float32 NPZ", "runtime": "Q8VNNI", "ZIP_MB": 8.556250, "host_us_per_row": 59.27, "members": 2, "heldout_status": "no matched final-mixture CV"}]
    mo.ui.table(serving_rows, selection=None, pagination=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### What the precision panel establishes
    The same student pair lost **0.000071** with Q8 relative to float32 and **0.001801** with 15-level rowwise-rounded weights relative to Q8, on 10 complete in-sample sequences. This measures fidelity of that pair. It does not reject a different larger compressed blend.

    ### Predeclared next complete endpoints
    Compare the feasible teacher4/5 Q8 baseline with: (1) teacher3/4/5 using FP16 disk transport and Q8 runtime; (2) student-half / teacher-trio-half using the same transport; (3) a larger mixed-precision teacher trio if needed. Keep normalization, input projection, biases and final head at float32. Measure matched held-out score, quantization fidelity, size, deterministic resets and one-CPU runtime together.

    Host timings above are runtime observations, not competition-container guarantees. We will not count an added member as useful merely because its errors correlate less.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### New measured storage screen
    Casting only recurrent matrices and head.0 to IEEE float16 **on disk** reduced the teacher-pair ZIP from **17.272 MB to 9.591 MB**, while retaining the existing loader and Q8VNNI backend. The compact pair shrank from12.914 to7.364MB. The shape-based raw ceiling for three wide teachers plus one narrow student is about18.46MB with header allowance, pending the actual third-member schema/build.

    This is approximate transport, not lossless storage: some Q8 codes and scales change. The next test must grade the larger endpoint itself. Q4 remains eligible if its total blend gain wins after its actual serving loss.
    """)
    return


app._unparsable_cell(
    r"""
    ready_top = json.loads((root / "submissions/10-11-67/READY_TOP_CANDIDATES_R2.json").read_text())
    ready_rows = [{"rank": c["rank"], "ZIP": c["path"].split("/")[-1], "MB": round(c["bytes"]/1e6,3), "role": c["role"], "precision": c["runtime_precision"], "actual held-out final score": "unavailable (all-data fitting)"} for c in ready_top["top_candidates"]]
    mo.vstack([mo.md("### Current reviewed provisional submissions"), mo.ui.table(ready_rows,selection=None)])
    """,
    column=None, disabled=False, hide_code=True, expand_output=False, name="Reviewed submissions"
)


app._unparsable_cell(
    """
    mo.md(\"\"\"
    ### Optimization priorities
    1. Confirm the **complete delivered blend** on identical source-qualified components, then cross-fold recipe confirmation.
    2. For weights, use a regularized covariance solution as initialization, followed by nonnegative simplex-constrained **SLSQP** on the exact clipped metric. Equal weights stay the frozen control. Existing covariance/scaling fits showed negligible gain.
    3. For model subsets and precision choices, use a small declared enumeration or beam search under **actual ZIP and one-CPU runtime** constraints. Q5–Q7 storage can unpack once into the existing byte kernel; it does not automatically speed inference.
    4. Short QAT with matched ordinary-fine-tuning/PTQ controls is proposed, not running. Existing QAT is weight-only Q8 and does not exactly simulate dynamic activation-Q8 deployment.

    All folds are exposed; component resampling is fixed-prediction uncertainty, not fresh CV or a public-score confidence interval. The full mathematical method and bounded policies are saved in `BLEND_OPTIMIZATION_METHODS_R1.md` and `MIXED_PRECISION_Q5_Q7_QAT_IDEAS_R1.md`.
    \"\"\")
    """,
    column=None, disabled=False, hide_code=True, expand_output=False, name="Optimization policy"
)


@app.cell(hide_code=True)
def _(mo, selector):
    mo.image(selector / "generated_r6/F1_BLEND_OUTPUT_EDA_R1.png")
    return


@app.cell(hide_code=True)
def _(json, mo, selector):
    native_grade = json.loads((selector / "native_f1_precision_panel_r1/artifacts/NATIVE_F1_OFFICIAL_GRADE_R1.json").read_text())
    native_rows = [{"endpoint": _k, "official_wp": _v["wp"], "t0": _v["t0"], "t1": _v["t1"]} for _k,_v in native_grade["candidate_scores"].items()]
    mo.vstack([mo.md("### Completed native precision comparison\n41 complete sequences / 14 whole components from exposed fold 1; fixed pool, 94,546 scored rows. This subset is not comparable to the full-fold score. Float16 storage minus float32 storage on the same Q8 trio: **−0.0000066413**. Prototype half-student / half-teacher-trio adds **+0.0008769294** here; the final H390 student's parent differs, so this is prototype evidence. The pair edges the trio on this small panel; full-fold float evidence favors the trio. Independent official-metric recomputation agrees."), mo.ui.table(native_rows, selection=None)])
    return


if __name__ == "__main__":
    app.run()
