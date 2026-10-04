"""Scout corpus fetcher: Kaggle competition writeups/discussions -> cached raw JSON + normalised JSONL.

Usage:
  python scripts/cr_scout_fetch.py list <competition> [--search TERM] [--pages N]
  python scripts/cr_scout_fetch.py fetch <competition> <topic_id> [<topic_id> ...]
  python scripts/cr_scout_fetch.py notebook <owner/slug>
  python scripts/cr_scout_fetch.py build            # rebuild corpus/docs.jsonl from raw cache

Rate limited to <= 1 request/s. Raw responses cached under corpus/raw/.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "research/campaign_2026-10-01/creative/cr_scout/corpus"
RAW = ROOT / "raw"
_last = [0.0]


def _throttle() -> None:
    dt = time.time() - _last[0]
    if dt < 1.1:
        time.sleep(1.1 - dt)
    _last[0] = time.time()


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def cmd_list(comp: str, search: str | None, pages: int) -> None:
    api = _api()
    out = RAW / comp
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for page in range(1, pages + 1):
        _throttle()
        r = api.competition_list_topics(comp, sort_by="top", page=page)
        got = [t.to_dict() for t in r.topics or []]
        rows += [{k: d.get(k) for k in ("id", "title", "votes", "commentCount", "topicUrl", "postDate")} for d in got]
        if not got:
            break
    (out / "topics_top.json").write_text(json.dumps(rows, indent=1))
    pat = re.compile(search or r"place|solution|writeup|\b\d+(st|nd|rd|th)\b", re.I)
    for row in rows:
        if not pat.search(row["title"] or "") and "/writeups/" not in (row["topicUrl"] or ""):
            continue
        print(f"{row['id']}\t{row['votes']}\t{row['title']}\t{row['topicUrl']}")


def cmd_fetch(comp: str, ids: list[int]) -> None:
    api = _api()
    out = RAW / comp
    out.mkdir(parents=True, exist_ok=True)
    for tid in ids:
        p = out / f"topic_{tid}.json"
        if p.exists():
            print("cached", p)
            continue
        _throttle()
        r = api.competition_list_topic_messages(comp, int(tid))
        p.write_text(json.dumps(r.to_dict(), default=str))
        print("saved", p, len(r.messages or []))


def cmd_notebook(ref: str) -> None:
    owner, slug = ref.split("/")
    out = RAW / "kernels" / slug
    if not any(out.glob("*.ipynb")):
        _throttle()
        subprocess.run(["kaggle", "kernels", "pull", ref, "-p", str(out), "-m"], check=True)
    print(out)


class _Blocks(HTMLParser):
    """Flatten writeup HTML into ordered markdown/code/image blocks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[dict] = []
        self.buf: list[str] = []
        self.in_pre = 0
        self.prefix = ""

    def _flush(self, kind: str = "markdown") -> None:
        text = "".join(self.buf).strip()
        if text:
            self.blocks.append({"type": kind, "text": text})
        self.buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "pre":
            self._flush()
            self.in_pre += 1
        elif tag in ("h1", "h2", "h3", "h4"):
            self._flush()
            self.buf.append("#" * int(tag[1]) + " ")
        elif tag in ("p", "div", "table", "blockquote"):
            self._flush()
        elif tag == "li":
            self.buf.append("\n- ")
        elif tag == "br":
            self.buf.append("\n")
        elif tag == "tr":
            self.buf.append("\n|")
        elif tag in ("td", "th"):
            self.buf.append(" ")
        elif tag == "code" and not self.in_pre:
            self.buf.append("`")
        elif tag == "img":
            self._flush()
            self.blocks.append({"type": "image", "src": a.get("src"), "alt": a.get("alt")})
        elif tag == "a" and a.get("href"):
            self.prefix = a["href"]

    def handle_endtag(self, tag):
        if tag == "pre":
            self.in_pre -= 1
            self._flush("code")
        elif tag in ("h1", "h2", "h3", "h4", "p", "ul", "ol", "table", "blockquote"):
            self._flush()
        elif tag in ("td", "th"):
            self.buf.append(" |")
        elif tag == "code" and not self.in_pre:
            self.buf.append("`")
        elif tag == "a" and self.prefix:
            if not self.prefix.startswith("#"):
                self.buf.append(f" <{self.prefix}>")
            self.prefix = ""

    def handle_data(self, data):
        self.buf.append(data if self.in_pre else re.sub(r"\s+", " ", data))

    def close(self):
        super().close()
        self._flush()


def html_blocks(s: str) -> list[dict]:
    p = _Blocks()
    p.feed(s or "")
    p.close()
    return p.blocks


def _comments(msgs: list[dict]) -> list[dict]:
    out = []
    for m in msgs:
        out.append(
            {
                "author": m.get("authorName") or (m.get("author") or {}).get("userName"),
                "votes": m.get("votes"),
                "date": m.get("postDate"),
                "text": "\n".join(b.get("text", "") for b in html_blocks(m.get("content", ""))),
                "replies": _comments(m.get("replies") or []),
            }
        )
    return out


def _nb_doc(path: Path, url: str) -> dict:
    nb = json.loads(path.read_text())
    blocks = []
    for c in nb["cells"]:
        src = "".join(c["source"])
        blocks.append({"type": "markdown" if c["cell_type"] == "markdown" else "code", "text": src})
        for name in c.get("attachments", {}):
            blocks.append({"type": "image", "src": f"attachment:{name}"})
        for o in c.get("outputs", []):
            if "text" in o:
                blocks.append({"type": "output", "text": "".join(o["text"])[:20000]})
            elif "data" in o:
                d = o["data"]
                if "text/plain" in d:
                    blocks.append({"type": "output", "text": "".join(d["text/plain"])[:20000]})
                if "image/png" in d:
                    blocks.append({"type": "image", "src": "inline:image/png"})
    meta = json.loads((path.parent / "kernel-metadata.json").read_text()) if (path.parent / "kernel-metadata.json").exists() else {}
    if meta.get("id"):
        url = f"https://www.kaggle.com/code/{meta['id']}"
    comps = meta.get("competition_sources") or [None]
    return {"url": url, "competition": comps[0], "title": meta.get("title", path.stem), "kind": "notebook", "blocks": blocks, "comments": []}


def cmd_build() -> None:
    docs = []
    for comp_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        index = {}
        for tf in comp_dir.glob("topics_*.json"):
            for row in json.loads(tf.read_text()):
                index[row["id"]] = row
        for f in sorted(comp_dir.glob("topic_*.json")):
            tid = int(f.stem.split("_")[1])
            msgs = json.loads(f.read_text()).get("messages") or []
            if not msgs:
                continue
            head, rest = msgs[0], msgs[1:]
            row = index.get(tid, {})
            docs.append(
                {
                    "url": "https://www.kaggle.com" + row["topicUrl"] if row.get("topicUrl") else f"{comp_dir.name}/topic/{tid}",
                    "competition": comp_dir.name,
                    "topic_id": tid,
                    "title": row.get("title"),
                    "votes": row.get("votes"),
                    "kind": "writeup",
                    "blocks": html_blocks(head.get("content", "")),
                    "comments": _comments(rest + (head.get("replies") or [])),
                }
            )
    for nbf in sorted(RAW.glob("**/*.ipynb")):
        docs.append(_nb_doc(nbf, f"https://www.kaggle.com/code/{nbf.stem}"))
    for jf in sorted(RAW.glob("web/*.json")):
        docs.append(json.loads(jf.read_text()))
    with open(ROOT / "docs.jsonl", "w") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(docs)} docs -> {ROOT / 'docs.jsonl'}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "list":
        search = a[a.index("--search") + 1] if "--search" in a else None
        pages = int(a[a.index("--pages") + 1]) if "--pages" in a else 1
        cmd_list(a[1], search, pages)
    elif a[0] == "fetch":
        cmd_fetch(a[1], [int(x) for x in a[2:]])
    elif a[0] == "notebook":
        cmd_notebook(a[1])
    elif a[0] == "build":
        cmd_build()
