#!/usr/bin/env python3
"""Download missing GeoTIFF files listed in missing_drive_files.json."""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
import re
import sys
import time
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_JSON = PROJECT_DIR / "missing_drive_files.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "missing_drive_tiffs"
CHUNK_SIZE = 1024 * 1024


def parse_drive_id(url: str) -> str:
    """Extract a Google Drive file id from a common sharing URL."""
    patterns = (
        r"/file/d/([^/]+)",
        r"[?&]id=([^&]+)",
        r"/uc\?[^#]*id=([^&]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not parse Google Drive file id from: {url}")


def safe_filename(name: str) -> str:
    name = os.path.basename(name.strip())
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name)
    if not name.lower().endswith((".tif", ".tiff")):
        raise ValueError(f"Not a TIFF filename: {name}")
    return name


def load_tiff_entries(json_path: Path) -> list[dict[str, str]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    entries: list[dict[str, str]] = []
    for item in data:
        filename = safe_filename(str(item.get("text", "")))
        href = str(item.get("href", ""))
        entries.append(
            {
                "file_id": parse_drive_id(href),
                "filename": filename,
                "href": href,
            }
        )
    return entries


def request_url(opener, url: str):
    return opener.open(
        Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36"
                )
            },
        ),
        timeout=120,
    )


def get_confirm_url(response, cookie_jar: CookieJar, file_id: str) -> str | None:
    for cookie in cookie_jar:
        if cookie.name.startswith("download_warning"):
            return (
                "https://drive.google.com/uc?"
                + urlencode(
                    {
                        "export": "download",
                        "confirm": cookie.value,
                        "id": file_id,
                    }
                )
            )

    page = response.read().decode("utf-8", errors="ignore")

    form_match = re.search(
        r'<form[^>]+id="download-form"[^>]+action="([^"]+)"[^>]*>(.*?)</form>',
        page,
        re.DOTALL,
    )
    if form_match:
        action = html.unescape(form_match.group(1))
        form_body = form_match.group(2)
        params = {}
        for name, value in re.findall(
            r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', form_body
        ):
            params[html.unescape(name)] = html.unescape(value)
        params.setdefault("id", file_id)
        params.setdefault("export", "download")
        return urljoin("https://drive.google.com", action) + "?" + urlencode(params)

    if "accounts.google.com" in page or "ServiceLogin" in page:
        raise RuntimeError("Google Drive requires login for this file")

    parsed = urlparse(response.url)
    query = parse_qs(parsed.query)
    if "confirm" in query:
        return response.url

    raise RuntimeError("Google Drive returned an HTML page instead of a file")


def open_drive_file(opener, cookie_jar: CookieJar, file_id: str):
    url = f"https://drive.google.com/uc?export=download&id={quote(file_id)}"
    response = request_url(opener, url)
    content_type = response.headers.get("Content-Type", "").lower()
    disposition = response.headers.get("Content-Disposition", "").lower()

    if "attachment" in disposition or "text/html" not in content_type:
        return response

    confirm_url = get_confirm_url(response, cookie_jar, file_id)
    if not confirm_url:
        raise RuntimeError("Could not get Google Drive confirmation URL")
    return request_url(opener, confirm_url)


def download_one(
    entry: dict[str, str],
    output_dir: Path,
    overwrite: bool,
    retries: int,
) -> tuple[str, str]:
    output_path = output_dir / entry["filename"]
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return entry["filename"], "skipped"

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            cookie_jar = CookieJar()
            opener = build_opener(HTTPCookieProcessor(cookie_jar))
            response = open_drive_file(opener, cookie_jar, entry["file_id"])
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise RuntimeError("received HTML instead of TIFF data")

            with tmp_path.open("wb") as f:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)

            if tmp_path.stat().st_size == 0:
                raise RuntimeError("downloaded file is empty")
            tmp_path.replace(output_path)
            return entry["filename"], "downloaded"
        except (HTTPError, URLError, TimeoutError, RuntimeError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)

    if tmp_path.exists():
        tmp_path.unlink()
    return entry["filename"], f"failed: {last_error}"


def iter_results(
    entries: Iterable[dict[str, str]],
    output_dir: Path,
    overwrite: bool,
    retries: int,
    workers: int,
):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_one, entry, output_dir, overwrite, retries)
            for entry in entries
        ]
        for future in concurrent.futures.as_completed(futures):
            yield future.result()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download TIFF files listed in missing_drive_files.json."
    )
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON, help="Input JSON file")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Local folder for downloaded TIFFs",
    )
    parser.add_argument("--workers", type=int, default=3, help="Parallel downloads")
    parser.add_argument("--retries", type=int, default=3, help="Retries per file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files that already exist locally",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be downloaded without downloading them",
    )
    args = parser.parse_args()

    entries = load_tiff_entries(args.json)

    print(f"Input JSON: {args.json}")
    print(f"Output dir: {args.out}")
    print(f"TIFF files: {len(entries)}")

    if args.dry_run:
        for entry in entries:
            print(f"{entry['filename']}: {entry['file_id']}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures: list[tuple[str, str]] = []
    for filename, status in iter_results(
        entries, args.out, args.overwrite, args.retries, max(1, args.workers)
    ):
        print(f"{filename}: {status}", flush=True)
        if status.startswith("failed"):
            counts["failed"] += 1
            failures.append((filename, status))
        else:
            counts[status] += 1

    print(
        "Done: "
        f"{counts['downloaded']} downloaded, "
        f"{counts['skipped']} skipped, "
        f"{counts['failed']} failed"
    )

    if failures:
        log_path = args.out / "download_failures.txt"
        with log_path.open("w", encoding="utf-8") as f:
            for filename, status in failures:
                f.write(f"{filename}\t{status}\n")
        print(f"Failure log: {log_path}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
