import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def timing_probe():

    import time
    from pathlib import Path
    t0 = time.perf_counter()
    time.sleep(2)
    p = Path('/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-05/independent_eda_20261006')
    p.mkdir(parents=True, exist_ok=True)
    (p / '_cell_timing_probe.txt').write_text(f'dt={time.perf_counter()-t0:.3f}\n')
    probe_done = True

    return (Path,)


@app.cell(hide_code=True)
def _(Path):

    import json
    root = Path("/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-06")
    report = json.loads((root / "wave1_anatomy.json").read_text())
    quiet = json.loads((root / "wave1_quiet.json").read_text())
    book = json.loads((root / "wave1_book.json").read_text())
    blend_wp = report["scores"]["blend_full_valid"]["wp"]
    class_wp = {k: v["wp"] for k, v in report["scores"]["blend_by_class"].items()}
    clocks = report["label_change_by_clock"]
    quiet_delta = quiet["delta_vs_incumbent"]
    mlp_delta = quiet["mlp_delta_vs_incumbent"]
    book_delta = book["delta_vs_incumbent"]
    class_share = quiet["class_weight_share"]

    return


if __name__ == "__main__":
    app.run()
