import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import runpy as _e535_runpy

    _e535_result = _e535_runpy.run_path(
        "research/campaign_2026-10-06/bandit_continuation/model_data_forensics_E535/E535_H230_OUTPUT_HOLD_DIAGNOSTIC_V1.py",
        run_name="__main__",
    )
    _e535_result["FIGURE"]
    return


if __name__ == "__main__":
    app.run()
