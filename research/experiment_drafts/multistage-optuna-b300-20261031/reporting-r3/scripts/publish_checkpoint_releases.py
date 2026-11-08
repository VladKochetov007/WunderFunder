"""Publish two verified checkpoint bundles as SemyonEpanov-owned GitHub releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from http.client import HTTPMessage
from pathlib import Path, PurePosixPath
from typing import BinaryIO

REPOSITORY = "VladKochetov007/WunderFunder"
API = "https://api.github.com"
ALLOWED_HOSTS = {
    "api.github.com",
    "uploads.github.com",
    "github.com",
    "release-assets.githubusercontent.com",
    "objects.githubusercontent.com",
}
SOURCE_COMMITS = {46: "50053909e365549b9146002d6c7d9118b81e9f97", 49: "61f0a00b"}
MODEL_COUNTS = {46: 8, 49: 16}
SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"


def sha256(path: Path) -> str:
    """Hash an archive without loading it all into RAM."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Keep the Git credential off redirected CDN requests."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: BinaryIO,
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
            raise RuntimeError("Unexpected release download redirect")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            redirected is not None
            and parsed.hostname != urllib.parse.urlparse(req.full_url).hostname
        ):
            redirected.remove_header("Authorization")
        return redirected


class GitHub:
    """Use the existing Git credential without writing or printing its value."""

    def __init__(self, root: Path) -> None:
        result = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=25,
            cwd=root,
            env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never"),
        )
        fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
        self.token = fields["password"]
        self.opener = urllib.request.build_opener(SafeRedirect())

    def request(self, method: str, path: str, payload: object = None) -> object:
        """Call one fixed-origin API endpoint and redact HTTP error bodies."""
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            API + path, data=data, headers=self.headers(), method=method
        )
        try:
            with self.opener.open(request, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"GitHub {method} failed with HTTP{error.code}: {path}") from None

    def headers(self) -> dict[str, str]:
        """Build an authentication header only for GitHub-owned API/upload endpoints."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "tabdata-checkpoint-publication",
        }

    def upload(self, release_id: int, path: Path) -> dict[str, object]:
        """Stream a bounded original asset with its exact Content-Length."""
        url = f"https://uploads.github.com/repos/{REPOSITORY}/releases/{release_id}/assets?"
        url += urllib.parse.urlencode({"name": path.name})
        headers = {
            **self.headers(),
            "Content-Type": "application/octet-stream",
            "Content-Length": str(path.stat().st_size),
        }
        with path.open("rb") as stream:
            request = urllib.request.Request(url, data=stream, headers=headers, method="POST")
            try:
                with self.opener.open(request, timeout=60) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                raise RuntimeError(
                    f"Asset upload failed with HTTP{error.code}: {path.name}"
                ) from None

    def download(self, asset: dict[str, object], destination: Path, digest: str) -> None:
        """Verify the actual downloadable bytes, stripping auth at CDN redirects."""
        url = f"{API}/repos/{REPOSITORY}/releases/assets/{asset['id']}"
        headers = {**self.headers(), "Accept": "application/octet-stream"}
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers=headers)
        with self.opener.open(request, timeout=60) as response, destination.open("xb") as target:
            for block in iter(lambda: response.read(1 << 20), b""):
                target.write(block)
        if destination.stat().st_size != asset["size"] or sha256(destination) != digest:
            raise ValueError("Downloaded release asset differs from the original")


def verify_archive(path: Path, expected_models: int) -> dict[str, object]:
    """Independently verify every manifest payload in an archive without extracting paths."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or any(
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or "\\" in name
            for name in names
        ):
            raise ValueError("Unsafe or duplicate archive entry")
        manifest = json.loads(archive.read("MANIFEST.json"))
        if (
            manifest.get("split_sha256") != SPLIT_SHA
            or hashlib.sha256(archive.read("shared/chronological_primary.json")).hexdigest()
            != SPLIT_SHA
        ):
            raise ValueError("Archive primary split differs from the authorized experiment")
        if len(manifest["runs"]) != expected_models:
            raise ValueError("Archive model count changed")
        for run in manifest["runs"]:
            entries = {entry["path"]: entry for entry in run["files"]}
            for role in run["roles"].values():
                if role["path"] not in entries:
                    raise ValueError("A checkpoint role has no hashed payload")
            for name, entry in entries.items():
                digest = hashlib.sha256()
                with archive.open(name) as stream:
                    for block in iter(lambda: stream.read(1 << 20), b""):
                        digest.update(block)
                if (
                    archive.getinfo(name).file_size != entry["bytes"]
                    or digest.hexdigest() != entry["sha256"]
                ):
                    raise ValueError(f"Archive member differs from its manifest: {name}")
        return manifest


def save(path: Path, payload: object) -> None:
    """Persist nonsecret delivery metadata for resumption and review."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Upload only the two locally verified model bundles and link their existing PRs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    assets_root = args.assets_root.resolve(strict=True)
    exports = json.loads((assets_root / "EXPORT_RECEIPT.json").read_text())["archives"]
    github = GitHub(root)
    user = github.request("GET", "/user")
    if user["login"] != "SemyonEpanov":
        raise RuntimeError("Checkpoint publication requires SemyonEpanov")
    repo_path = f"/repos/{REPOSITORY}"
    repository = github.request("GET", repo_path)
    if (
        repository["full_name"].lower() != REPOSITORY.lower()
        or not repository["permissions"]["push"]
        or repository.get("private") is not True
    ):
        raise RuntimeError("Repository publication permission is unavailable")
    published = []
    for export in exports:
        pr = int(export["pr"])
        if pr not in SOURCE_COMMITS:
            raise ValueError("Unexpected experiment PR")
        if export["model_count"] != MODEL_COUNTS[pr]:
            raise ValueError("Unexpected experiment model count")
        files = [assets_root / Path(export[key]).name for key in ("zip", "manifest", "checksums")]
        if sha256(files[0]) != export["zip_sha256"]:
            raise ValueError("Downloaded source archive hash changed")
        manifest = verify_archive(files[0], int(export["model_count"]))
        if manifest.get("pr") != pr:
            raise ValueError("Checkpoint archive belongs to another experiment PR")
        if json.loads(files[1].read_text()) != manifest:
            raise ValueError("External manifest differs from archive manifest")
        tag = f"tabdata-pr{pr}-checkpoints-20261029-r1"
        source = github.request("GET", repo_path + f"/commits/{SOURCE_COMMITS[pr]}")["sha"]
        releases = github.request("GET", repo_path + "/releases?per_page=100")
        release = next((entry for entry in releases if entry["tag_name"] == tag), None)
        if release is not None and (
            release["author"]["login"] != "SemyonEpanov" or release["target_commitish"] != source
        ):
            raise ValueError("Existing release belongs to another author or source revision")
        body = (
            f"Checkpoint artifacts for [experiment PR #{pr}](https://github.com/{REPOSITORY}/pull/{pr}).\n\n"
            f"{export['model_count']} models, selected at {manifest['inventory_asof_utc']}. "
            "Each checkpoint CPU-loads successfully and has finite model tensors. "
            "The manifest binds model, role, epoch, full-valid WP and original SHA-256.\n\n"
            "`best_observed` is exposed validation selection. `fixed_endpoint` preserves "
            "the predeclared TL3 epoch; `selected_runtime` preserves model.pt ancestry hashes. "
            "Checkpoints contain their original buffers/configuration; source histories and primary split are included.\n\n"
            f"Absent/untrained runs: {manifest['absent_runs']}. No new fit or validation scoring was performed."
        )
        if release is None:
            release = github.request(
                "POST",
                repo_path + "/releases",
                {
                    "tag_name": tag,
                    "target_commitish": source,
                    "draft": True,
                    "prerelease": True,
                    "make_latest": "false",
                    "name": f"tabdata PR{pr}: checkpoint artifacts",
                    "body": body,
                },
            )
        release_path = repo_path + f"/releases/{release['id']}"
        receipts = []
        for path in files:
            digest = sha256(path)
            current = github.request("GET", release_path + "/assets?per_page=100")
            asset = next((item for item in current if item["name"] == path.name), None)
            if asset is None:
                asset = github.upload(int(release["id"]), path)
            if asset["state"] != "uploaded" or asset["size"] != path.stat().st_size:
                raise ValueError("Release asset upload is incomplete")
            if asset.get("digest") not in (None, "sha256:" + digest):
                raise ValueError(
                    "GitHub asset digest differs; existing assets will not be overwritten"
                )
            destination = assets_root / "roundtrip" / str(pr) / path.name
            if destination.exists():
                if sha256(destination) != digest:
                    raise ValueError("Preserve mismatching prior roundtrip")
            else:
                github.download(asset, destination, digest)
            if path.suffix == ".zip":
                verify_archive(destination, int(export["model_count"]))
            receipts.append(
                {
                    "name": path.name,
                    "id": asset["id"],
                    "bytes": asset["size"],
                    "sha256": digest,
                    "url": asset["browser_download_url"],
                    "roundtrip_verified": True,
                }
            )
            save(
                assets_root / f"PR{pr}_UPLOAD_PROGRESS.json",
                {"release_id": release["id"], "assets": receipts},
            )
        release = github.request(
            "PATCH",
            release_path,
            {"draft": False, "prerelease": True, "make_latest": "false", "body": body},
        )
        # Draft assets use temporary untagged URLs; resolve their published URLs.
        published_assets = {
            asset["id"]: asset
            for asset in github.request("GET", release_path + "/assets?per_page=100")
        }
        for item in receipts:
            published_asset = published_assets[item["id"]]
            if published_asset["name"] != item["name"] or published_asset["size"] != item["bytes"]:
                raise ValueError("Published asset identity changed")
            item["url"] = published_asset["browser_download_url"]
        marker = f"<!-- tabdata-checkpoint-artifacts:{tag} -->"
        text = (
            marker
            + f"\nCheckpoint archives are now available: [PR{pr} release]({release['html_url']}).\n\n"
        )
        text += f"**{export['model_count']} models**, original unquantized checkpoints; each asset was downloaded again and SHA-256 verified.\n\n"
        text += "| Model | Best epoch | Best valid WP | Fixed TL3 endpoint |\n|---|---:|---:|---|\n"
        for run in manifest["runs"]:
            best = run["roles"]["best_observed"]
            fixed = run["roles"].get("fixed_endpoint")
            fixed_text = (
                "-" if fixed is None else f"epoch {fixed['epoch']}: {fixed['valid_wp']:.6f}"
            )
            text += (
                f"| `{run['run']}` | {best['epoch']} | {best['valid_wp']:.6f} | {fixed_text} |\n"
            )
        text += "\nAssets: " + " · ".join(f"[{item['name']}]({item['url']})" for item in receipts)
        text += f"\n\nAbsent/untrained: {manifest['absent_runs']}. Best-observed selection is exposed; fixed endpoints remain unchanged."
        comments_path = repo_path + f"/issues/{pr}/comments"
        comments = github.request("GET", comments_path + "?per_page=100")
        comment = next(
            (
                entry
                for entry in comments
                if marker in entry["body"] and entry["user"]["login"] == "SemyonEpanov"
            ),
            None,
        )
        if comment is None:
            comment = github.request("POST", comments_path, {"body": text})
        elif comment["body"] != text:
            comment = github.request(
                "PATCH", repo_path + f"/issues/comments/{comment['id']}", {"body": text}
            )
        result = {
            "pr": pr,
            "release": release["html_url"],
            "comment": comment["html_url"],
            "author": user["login"],
            "repository_private": repository["private"],
            "models": export["model_count"],
            "assets": receipts,
        }
        published.append(result)
        save(assets_root / "PUBLICATION_RECEIPT.json", published)
        print(
            json.dumps(
                {"published_pr": pr, "release": result["release"], "models": result["models"]}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
