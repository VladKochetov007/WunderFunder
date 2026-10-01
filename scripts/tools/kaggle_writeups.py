#!/usr/bin/env python3
"""Fetch Kaggle competition solution write-ups as markdown plus their images.

Anonymous: it uses the same JSON endpoints the kaggle.com web pages call, with a
session cookie from the home page. It never reads ~/.kaggle/kaggle.json.

    .venv/bin/python scripts/tools/kaggle_writeups.py TARGET [TARGET ...] \
        [--top 10] [--out research/kaggle] [--shots] [--comments 15] [--delay 2]

TARGET is one of
  * a competition slug, e.g. optiver-trading-at-the-close: fetches the write-ups
    linked from the private leaderboard for ranks <= --top, plus forum topics whose
    title looks like "Nth place" / "solution" with N <= --top;
  * a write-up URL  .../competitions/<slug>/writeups/<name>;
  * a discussion URL .../competitions/<slug>/discussion/<topic id>.

Output per topic: <out>/<slug>/<topic>/writeup.md (front matter + raw markdown with
image links rewritten to local files), comments.md (author replies and top comments),
images/ (downloaded originals), shots/ (with --shots: Playwright element screenshots of
the rendered figures and result tables). <out>/<slug>/INDEX.md lists what was saved.
Requests are serialized with at least --delay seconds between them and back off on
HTTP 429/5xx.
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.kaggle.com"
UA = "Mozilla/5.0 (X11; Linux x86_64) research-notes/1.0 (polite; low rate)"
RANK_RE = re.compile(r"(\d+)\s*(?:st|nd|rd|th)\b", re.I)
SOLUTION_RE = re.compile(r"place|solution|writeup|write-up", re.I)
IMG_MD_RE = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
IMG_HTML_RE = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", re.I)
IMG_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
           "image/webp": ".webp", "image/svg+xml": ".svg"}


class Kaggle:
    """Rate-limited anonymous client for kaggle.com's internal JSON API."""

    def __init__(self, delay: float):
        self.delay = delay
        self.last = 0.0
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.opener.addheaders = [("User-Agent", UA)]
        self._open(urllib.request.Request(BASE + "/")).read()

    def _open(self, req: urllib.request.Request, tries: int = 4):
        for k in range(tries):
            wait = self.last + self.delay - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            try:
                return self.opener.open(req, timeout=60)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or k == tries - 1:
                    raise
                retry = e.headers.get("Retry-After")
                time.sleep(float(retry) if retry and retry.isdigit() else 10.0 * 2**k)
        raise RuntimeError("unreachable")

    def api(self, method: str, payload: dict) -> dict:
        token = next((c.value for c in self.jar if c.name == "XSRF-TOKEN"), "")
        req = urllib.request.Request(
            f"{BASE}/api/i/{method}", data=json.dumps(payload).encode(),
            headers={"content-type": "application/json", "accept": "application/json",
                     "x-xsrf-token": token})
        return json.loads(self._open(req).read())

    def get_bytes(self, url: str) -> tuple[bytes, str]:
        with self._open(urllib.request.Request(url)) as r:
            return r.read(), r.headers.get_content_type()


def slugify(text: str, n: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n] or "topic"


def parse_target(t: str) -> tuple[str, str, str]:
    """Return (kind, competition slug, id) with kind in {comp, writeup, discussion}."""
    if not t.startswith("http"):
        return "comp", t.strip("/"), ""
    path = urllib.parse.urlparse(t).path.strip("/").split("/")
    if len(path) >= 4 and path[0] == "competitions" and path[2] in ("writeups", "discussion"):
        return ("writeup" if path[2] == "writeups" else "discussion"), path[1], path[3]
    raise SystemExit(f"unrecognized target: {t}")


def flatten_comments(comments: list[dict], depth: int = 0) -> list[tuple[int, dict]]:
    out = []
    for c in comments:
        out.append((depth, c))
        out.extend(flatten_comments(c.get("replies") or [], depth + 1))
    return out


def render_comments(comments: list[dict], keep: int) -> str:
    """Keep all threads where the topic author replied, then the top-voted rest."""
    def votes(c):
        return (c.get("votes") or {}).get("totalVotes", 0)

    def has_author(c):
        return any(x.get("authorType") == "TOPIC" for _, x in flatten_comments([c]))

    threads = sorted(comments, key=lambda c: (not has_author(c), -votes(c)))[:keep]
    lines = []
    for depth, c in flatten_comments(threads):
        who = (c.get("author") or {}).get("displayName", "?")
        tag = " (topic author)" if c.get("authorType") == "TOPIC" else ""
        body = (c.get("rawMarkdown") or "").strip().replace("\n", "\n" + "> " * (depth + 1))
        lines.append(f"{'> ' * (depth + 1)}**{who}**{tag}, votes {votes(c)}:\n"
                     f"{'> ' * (depth + 1)}{body}\n")
    return "\n".join(lines)


def save_images(k: Kaggle, md: str, html: str, img_dir: Path) -> tuple[str, list[str]]:
    urls = list(dict.fromkeys(IMG_MD_RE.findall(md) + IMG_HTML_RE.findall(html)))
    saved = []
    for i, u in enumerate(urls):
        full = urllib.parse.urljoin(BASE, u)
        if not full.startswith("http") or "/static/images/" in full:
            continue
        try:
            data, ctype = k.get_bytes(full)
        except Exception as e:  # keep going: a dead image link should not lose the text
            print(f"    image failed {full[:80]}: {e}", file=sys.stderr)
            continue
        ext = IMG_EXT.get(ctype) or Path(urllib.parse.urlparse(full).path).suffix[:5] or ".bin"
        img_dir.mkdir(parents=True, exist_ok=True)
        name = f"img_{i:02d}{ext}"
        (img_dir / name).write_bytes(data)
        md = md.replace(u, f"images/{name}")
        saved.append(name)
    return md, saved


def screenshot(url: str, shot_dir: Path, max_items: int = 6) -> list[str]:
    """Element screenshots of rendered content figures and result tables."""
    from playwright.sync_api import sync_playwright

    shot_dir.mkdir(parents=True, exist_ok=True)
    out = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1600}, user_agent=UA)
        page.goto(url, wait_until="networkidle", timeout=90_000)
        page.wait_for_timeout(1500)
        for sel, prefix, min_w in (("img", "fig", 250), ("table", "table", 0)):
            n = 0
            for el in page.query_selector_all(sel):
                if n >= max_items:
                    break
                if sel == "img":
                    src, w = el.evaluate("e => [e.src, e.naturalWidth]")
                    if w < min_w or "/static/images/" in src or "avatars" in src:
                        continue
                elif el.evaluate("e => e.rows.length") < 3:
                    continue
                el.scroll_into_view_if_needed()
                page.wait_for_timeout(300)
                name = f"{prefix}_{n:02d}.png"
                el.screenshot(path=str(shot_dir / name))
                out.append(name)
                n += 1
        browser.close()
    return out


def write_topic(k: Kaggle, args, slug: str, topic: str, meta: dict, md: str, html: str,
                comments: list[dict]) -> dict:
    d = Path(args.out) / slug / topic
    d.mkdir(parents=True, exist_ok=True)
    md, images = save_images(k, md, html, d / "images")
    front = "\n".join(f"{key}: {json.dumps(v, ensure_ascii=False)}" for key, v in meta.items())
    (d / "writeup.md").write_text(f"---\n{front}\n---\n\n# {meta['title']}\n\n{md}\n")
    if comments and args.comments:
        (d / "comments.md").write_text(
            f"# Comments on {meta['title']}\n\n{render_comments(comments, args.comments)}")
    shots = screenshot(meta["url"], d / "shots") if args.shots else []
    print(f"  saved {d} ({len(md)} chars, {len(images)} images, {len(shots)} shots)")
    return {**meta, "dir": str(d.relative_to(Path(args.out) / slug)),
            "images": len(images), "shots": len(shots)}


def fetch_writeup(k: Kaggle, args, slug: str, wslug: str, rank=None) -> dict:
    w = k.api("discussions.WriteUpsService/GetWriteUpBySlug",
              {"slug": wslug, "competitionName": slug})
    m = w.get("message") or {}
    comments = []
    if args.comments and w.get("topicId"):
        comments = k.api("discussions.DiscussionsService/GetForumMessagesInTopic",
                         {"topicId": w["topicId"], "newUserSort": "COMMENT_LIST_SORT_BY_HOT",
                          "filter": "ALL", "includeFirstForumMessage": False}).get("comments", [])
    meta = {"title": w.get("title", wslug), "url": BASE + w.get("url", ""),
            "authors": w.get("authors", ""), "rank": rank or m.get("competitionRanking"),
            "date": (w.get("publishTime") or m.get("postDate") or "")[:10],
            "votes": (m.get("votes") or {}).get("totalVotes"), "topic_id": w.get("topicId")}
    return write_topic(k, args, slug, wslug, meta, m.get("rawMarkdown", ""),
                       m.get("content", ""), comments)


def fetch_discussion(k: Kaggle, args, slug: str, topic_id: str, rank=None) -> dict:
    t = k.api("discussions.DiscussionsService/GetForumTopicById",
              {"forumTopicId": int(topic_id), "includeComments": True})["forumTopic"]
    m = t.get("firstMessage") or {}
    meta = {"title": t.get("name", topic_id),
            "url": f"{BASE}/competitions/{slug}/discussion/{topic_id}",
            "authors": t.get("authorUserDisplayName", ""),
            "rank": rank or m.get("competitionRanking"), "date": (t.get("postDate") or "")[:10],
            "votes": t.get("totalVotes"), "topic_id": int(topic_id)}
    return write_topic(k, args, slug, f"d{topic_id}-{slugify(meta['title'], 40)}", meta,
                       m.get("rawMarkdown", ""), m.get("content", ""), t.get("comments") or [])


def discover(k: Kaggle, args, slug: str) -> list[tuple[str, str, int | None]]:
    """Solution topics for a competition: leaderboard write-up links, then forum titles."""
    comp = k.api("competitions.CompetitionService/GetCompetition", {"competitionName": slug})
    found: dict[str, tuple[str, str, int | None]] = {}
    lb = k.api("competitions.LeaderboardService/GetLeaderboard",
               {"competitionId": comp["id"], "leaderboardMode": "LEADERBOARD_MODE_DEFAULT"})
    teams = {t["teamId"]: t for t in lb.get("teams", [])}
    for row in lb.get("privateLeaderboard") or lb.get("publicLeaderboard") or []:
        if row["rank"] > args.top:
            continue
        url = (teams.get(row["teamId"]) or {}).get("solutionWriteUpUrl") or ""
        if url:
            kind, _, ident = parse_target(urllib.parse.urljoin(BASE, url))
            found[f"{kind}:{ident}"] = (kind, ident, row["rank"])
    for query, page in ((q, pg) for q in args.search for pg in range(1, args.forum_pages + 1)):
        topics = k.api("discussions.DiscussionsService/GetTopicListByForumId", {
            "category": "TOPIC_LIST_CATEGORY_ALL", "group": "TOPIC_LIST_GROUP_ALL",
            "customGroupingIds": [], "author": "TOPIC_LIST_AUTHOR_UNSPECIFIED",
            "myActivity": "TOPIC_LIST_MY_ACTIVITY_UNSPECIFIED",
            "recency": "TOPIC_LIST_RECENCY_UNSPECIFIED", "filterCategoryIds": [],
            "searchQuery": query, "sortBy": "TOPIC_LIST_SORT_BY_TOP", "page": page,
            "forumId": comp["forumId"]}).get("topics", [])
        for t in topics:
            title = t.get("title", "")
            r = RANK_RE.search(title)
            rank = int(r.group(1)) if r else None
            if not SOLUTION_RE.search(title) or (rank or 0) > args.top:
                continue
            kind, _, ident = parse_target(urllib.parse.urljoin(BASE, t["topicUrl"]))
            found.setdefault(f"{kind}:{ident}", (kind, ident, rank))
    return sorted(found.values(), key=lambda x: (x[2] is None, x[2] or 0))


def write_index(out: Path, slug: str, rows: list[dict]) -> None:
    path = out / slug / "INDEX.md"
    old = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("| ") and "](" in line:
                old[line.split("|")[4].strip()] = line
    for r in rows:
        old[f"[{r['dir']}]({r['dir']}/writeup.md)"] = (
            f"| {r['rank'] or '-'} | {r['title'].replace('|', '/')} | {r['authors']} | "
            f"[{r['dir']}]({r['dir']}/writeup.md) | {r['images']} | {r['shots']} | {r['url']} |")
    lines = sorted(old.values(), key=lambda s: (not s.split("|")[1].strip().isdigit(),
                                                int(s.split("|")[1]) if s.split("|")[1].strip().isdigit() else 0))
    path.write_text(f"# {slug}: saved write-ups\n\n| rank | title | authors | dir | images | shots | url |\n"
                    "|---|---|---|---|---|---|---|\n" + "\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+")
    ap.add_argument("--top", type=int, default=10, help="max private-LB rank for a competition slug")
    ap.add_argument("--out", default="research/kaggle")
    ap.add_argument("--shots", action="store_true", help="Playwright screenshots of figures/tables")
    ap.add_argument("--comments", type=int, default=15, help="comment threads kept (0 = none)")
    ap.add_argument("--delay", type=float, default=2.0, help="min seconds between requests")
    ap.add_argument("--search", nargs="*", default=["", "solution", "place"],
                    help="forum search queries for solution topics ('' = top topics)")
    ap.add_argument("--forum-pages", type=int, default=2, help="forum pages (20 topics) per query")
    ap.add_argument("--list", action="store_true", help="only list discovered topics")
    args = ap.parse_args()

    k = Kaggle(args.delay)
    for target in args.targets:
        kind, slug, ident = parse_target(target)
        todo = discover(k, args, slug) if kind == "comp" else [(kind, ident, None)]
        print(f"{slug}: {len(todo)} topic(s)")
        rows = []
        for kd, ident, rank in todo:
            print(f"  {kd} {ident} rank={rank}")
            if args.list:
                continue
            try:
                fn = fetch_writeup if kd == "writeup" else fetch_discussion
                rows.append(fn(k, args, slug, ident, rank))
            except urllib.error.HTTPError as e:
                print(f"  failed {kd} {ident}: HTTP {e.code}", file=sys.stderr)
        if rows:
            write_index(Path(args.out), slug, rows)


if __name__ == "__main__":
    main()
