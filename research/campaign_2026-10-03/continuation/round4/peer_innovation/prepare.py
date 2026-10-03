from __future__ import annotations

import hashlib
import json

from common import PRIOR, ROOT, dump, sha


def main() -> None:
    panel = json.loads((PRIOR / "samples.json").read_text())
    components = sorted(
        [s["component"] for s in panel["samples"]],
        key=lambda c: hashlib.sha256(f"peer-innovation-split-v1:{c}".encode()).digest(),
    )
    assert len(set(components)) == 24
    panel.update(
        fit_components=components[:16],
        test_components=components[16:],
        selection="frozen metadata component split, exact old origin panel; no new mask/class/Y read",
        prior_sample_sha256=sha(PRIOR / "samples.json"),
    )
    dump("samples.json", panel)
    dump(
        "frozen_sources.json",
        {
            f: sha(ROOT / f)
            for f in ["PREREG.json", "common.py", "extract.py", "prepare.py", "samples.json"]
        },
    )
    print(
        json.dumps(
            dict(
                fit_components=components[:16],
                test_components=components[16:],
                queries=sum(s["m_selected"] for s in panel["samples"]),
            )
        )
    )


if __name__ == "__main__":
    main()
