"""Install reviewed proposal packets on isolated branches; never run a model fit."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN = ROOT / "research/campaign_2026-10-17/research_reset_r1"
WORKTREE = ROOT.parent / "WunderFunder-rr17-packets"
BASE = "ed40bea53b66f65a437f9ea3658a982ebdbe5251"


def git(*args: str, cwd: Path = ROOT) -> str:
    """Run a checked Git command without a shell and return compact output."""
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def install(packet_id: str) -> dict[str, object]:
    """Create one proposal branch from the frozen research-team integration base."""
    branch = f"experiment/{packet_id}"
    git("switch", "-c", branch, BASE, cwd=WORKTREE)
    target = WORKTREE / "research/experiment_drafts" / packet_id
    shutil.copytree(CAMPAIGN / "drafts" / packet_id, target)
    canonical = "../../campaign_2026-10-17/research_reset_r1/IDEAS_25.md"
    for path in target.glob("*.md"):
        text = path.read_text().replace("../../IDEAS_25.md", canonical)
        text = text.replace(
            "research-team SHA, experiment source SHA, PR URL and merge SHA are null until the parent creates a clean branch.",
            f"Research-team base is `{BASE}`; packet source SHA is recorded in the publication receipt. PR and merge remain absent.",
        )
        text += (
            f"\n## Installed branch status\n\nInstalled on `{branch}` from research-team `{BASE}`. "
            "This remains a proposal with no executable training adapter. The original staging "
            "description above is preserved as preparation history; actual publication and "
            "commit identity are recorded in the campaign publication receipt. No PR exists "
            "until an authenticated API session creates it.\n"
        )
        path.write_text(text)
    recipe_path = target / "recipe.json"
    recipe = json.loads(recipe_path.read_text())
    recipe["owner"] = "root controller; implementation/fit owner unassigned"
    recipe["ancestry"]["research_team_base_sha"] = BASE
    recipe["canonical_idea"]["staging_source_path"] = canonical
    recipe["publication"] = {
        "branch": branch,
        "source_commit_binding": "publication/BRANCHES.json records this packet commit",
        "pr_url": None,
        "status": "proposal_only",
    }
    recipe_path.write_text(json.dumps(recipe, indent=2, ensure_ascii=False) + "\n")
    title = recipe["canonical_idea"]["title"]
    index = WORKTREE / "research/unified/ideas/README.md"
    with index.open("a") as handle:
        handle.write(
            f"\n- [{packet_id}: {title}](../../experiment_drafts/{packet_id}/README.md) "
            f"— H-{packet_id.upper()}/E-{packet_id.upper()}, proposal only; independent critique "
            "and exact30paired cells/arm are specified, adapters/data/ancestry remain pending, "
            "training unauthorized and no measured gain.\n"
        )
    parent_index = target.parent / "README.md"
    if not parent_index.exists():
        parent_index.write_text("# Experiment proposal packets\n")
    with parent_index.open("a") as handle:
        handle.write(f"\n- [{packet_id}]({packet_id}/README.md) — {title}; proposal only, no fit.\n")
    git("add", str(target), str(index), str(parent_index), cwd=WORKTREE)
    git("diff", "--cached", "--check", cwd=WORKTREE)
    git("commit", "-m", f"Prepare {packet_id} no-fit experiment proposal", cwd=WORKTREE)
    commit = git("rev-parse", "HEAD", cwd=WORKTREE)
    return {
        "id": packet_id,
        "branch": branch,
        "commit": commit,
        "base": BASE,
        "pr_url": None,
        "files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(target.iterdir())
        },
    }


def main() -> None:
    """Prepare25 isolated proposal commits and write their local identity receipt."""
    if WORKTREE.exists():
        raise RuntimeError("Refusing to reuse an existing packet worktree")
    git("worktree", "add", "--detach", "--no-checkout", str(WORKTREE), BASE)
    git("sparse-checkout", "init", "--cone", cwd=WORKTREE)
    git(
        "sparse-checkout", "set", "research/campaign_2026-10-17", "research/unified/ideas",
        "research/experiment_drafts", cwd=WORKTREE,
    )
    git("checkout", "--detach", BASE, cwd=WORKTREE)
    records = []
    output = CAMPAIGN / "publication/BRANCHES.json"
    for number in range(1, 26):
        records.append(install(f"rr17-{number:02}"))
        output.write_text(json.dumps({
            "base": BASE, "worktree": str(WORKTREE), "published": False,
            "training_authorized": False, "branches": records,
        }, indent=2) + "\n")
        print(f"Prepared {records[-1]['branch']} {records[-1]['commit'][:12]}", flush=True)


if __name__ == "__main__":
    main()
