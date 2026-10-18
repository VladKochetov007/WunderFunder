"""Exercise uncertain acknowledgments, author ownership and comparison weighting."""

from __future__ import annotations

import pytest
from github_api import GitHub
from publish_updates import publish_snapshot
from render_report import marker, summary_table


def snapshot() -> dict:
    """Return a small scored cell with an actual-shaped immutable receipt identity."""
    return dict(
        experiment_id="PR15",
        expected_cells=120,
        expected_per_arm=30,
        completed=[
            dict(id="cell", arm="baseline", fold=0, seed=1001, wp=0.6, receipt_sha256="a" * 64)
        ],
        active=16,
        pending=103,
        errors=[],
        source={},
    )


class FakeGitHub:
    """Model a successful write whose acknowledgment can be lost in transport."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.fail_after_accept = False
        self.patched: list[int] = []

    def request(self, method: str, suffix: str) -> dict:
        repo = dict(full_name="VladKochetov007/WunderFunder")
        return dict(base=dict(ref="research-team", repo=repo),
                    head=dict(ref="experiment/geometry", repo=repo), state="open")

    def comments(self, number: int) -> list[dict]:
        return list(self.rows)

    def post(self, number: int, body: str) -> dict:
        row = dict(
            id=len(self.rows) + 1,
            html_url=f"https://example.test/{len(self.rows) + 1}",
            body=body,
            user=dict(login="owner"),
        )
        self.rows.append(row)
        if self.fail_after_accept:
            self.fail_after_accept = False
            raise RuntimeError("Acknowledgment lost after the server accepted the write")
        return row

    def patch(self, comment_id: int, body: str) -> dict:
        self.patched.append(comment_id)
        row = next(row for row in self.rows if row["id"] == comment_id)
        row["body"] = body
        return row


TARGET = dict(
    number=15, head_ref="experiment/geometry", evidence_url="https://example.test/evidence"
)


def test_uncertain_write_is_reconciled_without_duplicate_result() -> None:
    """The next pass finds the accepted marker rather than POSTing it again."""
    client = FakeGitHub()
    client.fail_after_accept = True
    state = {}
    with pytest.raises(RuntimeError, match="Acknowledgment lost"):
        publish_snapshot(client, snapshot(), TARGET, state, "owner", "completion")
    publish_snapshot(client, snapshot(), TARGET, state, "owner", "completion")
    publish_snapshot(client, snapshot(), TARGET, state, "owner", "hourly")
    assert sum(":results:" in row["body"] for row in client.rows) == 1
    assert state["endpoints"] == {"cell": "a" * 64}


def test_copied_status_marker_does_not_authorize_editing_teammate() -> None:
    """A status marker alone cannot grant ownership of someone else's comment."""
    client = FakeGitHub()
    client.rows.append(
        dict(
            id=1,
            html_url="https://example.test/foreign",
            body=marker("PR15", "status"),
            user=dict(login="teammate"),
        )
    )
    publish_snapshot(client, snapshot(), TARGET, {}, "owner", "hourly")
    assert 1 not in client.patched
    assert client.rows[0]["user"]["login"] == "teammate"


def test_mean_gives_folds_equal_weight_in_unbalanced_matrix() -> None:
    """Two seeds in one fold must not outweigh each other represented fold."""
    rows = [
        dict(id="a", arm="wide", fold=0, seed=1003, wp=0.0),
        dict(id="b", arm="wide", fold=0, seed=1004, wp=1.0),
    ]
    rows += [dict(id=f"f{f}", arm="wide", fold=f, seed=1003, wp=0.0) for f in [1, 2, 3]]
    table = summary_table(dict(experiment_id="E-B300-R01", expected_per_arm=30, completed=rows))
    assert "0.12500000" in table
    assert "0.20000000" not in table


def test_client_rejects_another_repository_before_network() -> None:
    """The unattended client cannot accidentally post to a different project."""
    with pytest.raises(ValueError, match="outside"):
        GitHub("test-credential").request("POST", "/repos/another/project/issues/1/comments", {})


def test_lost_acknowledgment_survives_changed_batch_membership() -> None:
    """A newly completed earlier-sorted cell cannot duplicate an accepted endpoint."""
    client = FakeGitHub()
    client.fail_after_accept = True
    state = {}
    with pytest.raises(RuntimeError):
        publish_snapshot(client, snapshot(), TARGET, state, "owner", "completion")
    changed = snapshot()
    changed["completed"].append(
        dict(id="aaa", arm="moments", fold=0, seed=1001, wp=0.61, receipt_sha256="b" * 64)
    )
    publish_snapshot(client, changed, TARGET, state, "owner", "completion")
    result_bodies = [row["body"] for row in client.rows if ":results:" in row["body"]]
    assert len(result_bodies) == 2
    assert sum('"id": "cell"' in body for body in result_bodies) == 1
