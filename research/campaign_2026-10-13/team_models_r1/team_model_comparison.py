import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path
    import marimo as mo
    import pandas as pd

    root = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
    campaign = root / "research/campaign_2026-10-13/team_models_r1"
    return campaign, json, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Team kotiki — model and inference comparison
    Shared code and weights are authorized. Final-model supplied-validation scores are **diagnostic**, not clean CV. Reported public: pair_ftt2 **0.6682**, xteam_h4 **0.6681**, E **0.6675**. Latest user report: the new seed6 cross-team blend scored **0.6679** (46m51s/650MB); the later E upload is confirmed at **0.6675**.
    """)
    return


@app.cell(hide_code=True)
def _(campaign, json, mo, pd):
    portal = json.loads((campaign / "public_results.json").read_text())
    _latest_portal = json.loads((campaign / "PUBLIC_RESULTS_20261013_R2.json").read_text())
    _portal_by_id = {row["portal_id"]: row for row in portal["records"]}
    _portal_by_id.update({row["portal_id"]: row for row in _latest_portal["records"]})
    portal["records"] = list(_portal_by_id.values())
    portal_table = pd.DataFrame(portal["records"])
    portal_table["minutes"] = portal_table["duration_seconds"] / 60
    mo.md("## Reported public outcomes\nUser-transcribed portal evidence; filenames are not portal-verified hashes. Pending durations are elapsed time and timeouts are censored.")

    return (portal_table,)


@app.cell(hide_code=True)
def _(mo, portal_table):
    mo.ui.table(portal_table[["filename", "score", "status", "minutes", "portal_id"]], selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Target CPU: AMD EPYC 9224
    **xteam_h4 already uses our exact Rust libraries.** Both engines contain AVX-512 VNNI paths. Zen 4 uses 256-bit internal paths and 1 MB L2 per core; target-device dispatch and same-weight paired timing remain unmeasured.

    The 44:42 pair, 49:16 E and 50:16 xteam portal durations compare different models. Local Intel profiles cannot decide a host winner. See the [saved runtime reflection](https://github.com/VladKochetov007/WunderFunder/blob/main/research/campaign_2026-10-13/team_models_r1/EPYC_RUNTIME_REFLECTION_R1.md) for evidence and proposed controls.
    """)
    return


@app.cell(hide_code=True)
def _(campaign, json, mo, pd):
    traffic = json.loads((campaign / "LOGICAL_Q8_WEIGHT_TRAFFIC_R1.json").read_text())
    traffic_table = pd.DataFrame([{key: value for key, value in row.items() if key != "matrix_bytes"} for row in traffic["rows"]])
    mo.vstack([mo.md("## Matrix payload and cache pressure\nDerived own-Q8 unpadded weight bytes. This excludes scales, padding, scratch and retained copies; it is not measured cache traffic or the C++ layout."), mo.ui.table(traffic_table, selection=None)])
    return


@app.cell(hide_code=True)
def _(campaign, json, mo, pd):
    inventory = json.loads((campaign / "INVENTORY.json").read_text())
    archive_rows = [{"package": _name, "MB": _p["bytes"]/1e6, "members": len(_p["members"]), "sha256": _p["sha256"], "scope": _p["scope"]} for _name, _p in inventory["packages"].items()]
    mo.ui.table(pd.DataFrame(archive_rows), selection=None)
    return (inventory,)


@app.cell(hide_code=True)
def _(inventory, mo, pd):
    weight_sources = {}
    for _name, _p in inventory["packages"].items():
        for _file, _v in _p["files"].items():
            if _file.endswith(".npz"):
                weight_sources.setdefault(_v["sha256"], []).append(f"{_name}/{_file}")
    duplicate_rows = [{"weight_sha256": _sha, "shared_by": "; ".join(_paths)} for _sha, _paths in weight_sources.items() if len(_paths)>1]
    mo.ui.table(pd.DataFrame(duplicate_rows), selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    refresh = mo.ui.refresh(options=[30, 60, 120], default_interval=60)
    refresh
    return (refresh,)


@app.cell(hide_code=True)
def _(campaign, json, mo, pd, refresh):
    refresh
    progress_rows = []
    for _path in sorted((campaign / "scores").glob("*/progress.jsonl")):
        _lines = _path.read_text().splitlines()
        if _lines:
            _last = json.loads(_lines[-1])
            progress_rows.append({"run": _path.parent.name, "completed sequences": _last["completed"], "total": _last["total"], "wall minutes": _last["wall_seconds"] / 60, "updated UTC": _last["utc"]})
    mo.ui.table(pd.DataFrame(progress_rows), selection=None) if progress_rows else mo.md("No replay progress yet.")
    return


@app.cell(hide_code=True)
def _(campaign, json, mo, pd, refresh):
    refresh
    bank_rows = []
    for _path in sorted((campaign / "member_bank").glob("*/progress.jsonl")):
        _lines = _path.read_text().splitlines()
        if _lines:
            _last = json.loads(_lines[-1])
            bank_rows.append({"member": _path.parent.name, "completed sequences": _last["completed"], "total": _last["total"], "wall seconds": _last["wall_seconds"], "terminal receipt": (_path.parent / "RESULT.json").exists()})
    mo.vstack([mo.md("## Individual FP32 members\nDiagnostic exports; native serving precision is evaluated separately."), mo.ui.table(pd.DataFrame(bank_rows), selection=None)]) if bank_rows else mo.md("Member exports have not started.")
    return


@app.cell(hide_code=True)
def _(campaign, json, mo, pd, refresh):
    refresh
    result_rows = []
    for _path in sorted((campaign / "scores").glob("*/RESULT.json")):
        _r = json.loads(_path.read_text())
        result_rows.append({"package": _r["candidate"], "mode": _r["mode"], "groups": len(_r["groups"]), "WP diagnostic": _r["score"]["wp"], "t0": _r["score"]["t0"], "t1": _r["score"]["t1"], "seconds": _r["wall_seconds"]})
    mo.ui.table(pd.DataFrame(result_rows), selection=None) if result_rows else mo.md("Full scoring has not completed. Short profiles are serving checks, not leaderboard forecasts.")
    return


@app.cell(hide_code=True)
def _(campaign, json, mo, pd):
    member_grade = json.loads((campaign / "scores/member_bank_fp32_r1/RESULT.json").read_text())
    member_grade_table = pd.DataFrame([
        {"endpoint": name, "diagnostic_wp": result["wp"], "t0": result["t0"], "t1": result["t1"]}
        for name, result in member_grade["grades"].items()
    ]).sort_values("diagnostic_wp", ascending=False)
    mo.vstack([
        mo.md("## Complete FP32 comparison\n14 members and 9 fixed mixtures; 1,873 sequences. **Final-fit/unknown validation overlap: diagnostic scores, not clean CV or public forecasts.** The new seed-6 submission is not among these nine mixtures."),
        mo.ui.table(member_grade_table),
    ])
    return


@app.cell(hide_code=True)
def _(campaign, json, mo):
    team_release = json.loads((campaign.parents[2] / "submissions/10-13-team-r1/RELEASE_LOCATION_R2.json").read_text())
    mo.md(f"""## New cross-team submission ready
    `{team_release['zip_path']}`

    Equal weights: teammate distilled seed6, teammate FTT seed6, our Z4, our H390 seed3. Uses our optimized Rust Q8/VNNI runtime, byte-identical to E.

    ZIP: {team_release['zip_bytes']:,} bytes. ABI, isolated Python3.11 import, and complete-sequence V2/V3 reset checks passed. This is an exploratory public probe; user-reported public score is **0.6679**,46m51s/650MB (portal0mbzevhv).
    """)
    return


if __name__ == "__main__":
    app.run()
