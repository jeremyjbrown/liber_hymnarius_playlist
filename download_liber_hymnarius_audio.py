#!/usr/bin/env python3
"""
Download every MP3 exposed by LiberHymnarius.org's MediaWiki file API.

Default behavior:
  - Enumerate all uploaded files through MediaWiki's `allimages` API.
  - Keep MP3/audio files only.
  - Download them into ./audio/
  - Skip an existing file when its byte size matches the wiki's current file.
  - Write ./audio/manifest.csv with source metadata.
  - Download atomically through a temporary .part file.
  - Retry transient network errors.

Examples:
    python scripts/download_liber_hymnarius_audio.py
    python scripts/download_liber_hymnarius_audio.py --dry-run
    python scripts/download_liber_hymnarius_audio.py --output audio/liber-hymnarius
    python scripts/download_liber_hymnarius_audio.py --force

This script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_API = "https://liberhymnarius.org/api.php"
DEFAULT_OUTPUT = Path("audio")
DEFAULT_MANIFEST = "manifest.csv"
USER_AGENT = (
    "LiberHymnariusPlaylistDownloader/1.0 "
    "(personal archival/listening tool; polite MediaWiki client)"
)

# Characters Windows does not permit in filenames. The script remains portable
# if the repository is cloned on Windows later.
WINDOWS_RESERVED = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class AudioFile:
    original_name: str
    url: str
    description_url: str
    size: int | None
    mime: str
    timestamp: str

    @property
    def local_name(self) -> str:
        return safe_filename(self.original_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all MP3 recordings from LiberHymnarius.org."
    )
    parser.add_argument(
        "--api",
        default=DEFAULT_API,
        help=f"MediaWiki API endpoint (default: {DEFAULT_API})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"Manifest filename inside output directory (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Delay in seconds between downloads (default: 0.35)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts for each HTTP request (default: 3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even when an existing file has the expected size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files and total size without downloading them.",
    )
    return parser.parse_args()


def safe_filename(name: str) -> str:
    """Return a cross-platform filename while preserving readable Latin text."""
    name = WINDOWS_RESERVED.sub("_", name)
    name = name.rstrip(" .")
    if not name:
        raise ValueError("MediaWiki returned an empty/invalid filename")
    return name


def human_size(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def get_json(
    base_url: str,
    params: dict[str, str | int],
    retries: int,
) -> dict:
    query = urlencode(params)
    url = f"{base_url}?{query}"

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == retries:
                break
            wait = min(2 ** (attempt - 1), 8)
            print(
                f"API request failed ({exc}); retrying in {wait}s...",
                file=sys.stderr,
            )
            time.sleep(wait)

    raise RuntimeError(f"Could not query MediaWiki API: {last_error}")


def enumerate_audio_files(api_url: str, retries: int) -> Iterator[AudioFile]:
    """
    Enumerate all MP3/audio uploads using MediaWiki's allimages list.

    `allimages` is paginated. MediaWiki returns continuation parameters that
    must be supplied verbatim on the next request.
    """
    continuation: dict[str, str] = {}

    while True:
        params: dict[str, str | int] = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "list": "allimages",
            "ailimit": "max",
            "aiprop": "timestamp|url|size|mime",
        }
        params.update(continuation)

        payload = get_json(api_url, params, retries)
        items = payload.get("query", {}).get("allimages", [])

        for item in items:
            name = str(item.get("name", ""))
            mime = str(item.get("mime", "")).lower()

            # Some older MediaWiki installations report audio/mpeg, others
            # audio/mp3. Filename filtering provides a conservative fallback.
            if not (
                name.lower().endswith(".mp3")
                or mime in {"audio/mpeg", "audio/mp3", "audio/x-mp3"}
            ):
                continue

            url = str(item.get("url", ""))
            if not url:
                print(
                    f"Warning: no download URL returned for {name!r}; skipping.",
                    file=sys.stderr,
                )
                continue

            raw_size = item.get("size")
            try:
                size = int(raw_size) if raw_size is not None else None
            except (TypeError, ValueError):
                size = None

            yield AudioFile(
                original_name=name,
                url=url,
                description_url=str(item.get("descriptionurl", "")),
                size=size,
                mime=str(item.get("mime", "")),
                timestamp=str(item.get("timestamp", "")),
            )

        next_continue = payload.get("continue")
        if not next_continue:
            break

        continuation = {
            str(key): str(value)
            for key, value in next_continue.items()
        }


def unique_local_names(files: list[AudioFile]) -> dict[str, str]:
    """
    Map source filenames to collision-safe local filenames.

    Sanitization can theoretically make two different source filenames equal.
    If that occurs, append a numeric suffix instead of overwriting a file.
    """
    used: set[str] = set()
    result: dict[str, str] = {}

    for audio in files:
        candidate = audio.local_name
        stem = Path(candidate).stem
        suffix = Path(candidate).suffix
        counter = 2

        while candidate.casefold() in used:
            candidate = f"{stem}-{counter}{suffix}"
            counter += 1

        used.add(candidate.casefold())
        result[audio.original_name] = candidate

    return result


def download_file(
    audio: AudioFile,
    destination: Path,
    retries: int,
    force: bool,
) -> str:
    """
    Download one file.

    Returns one of: "downloaded", "skipped".
    """
    if (
        not force
        and destination.exists()
        and audio.size is not None
        and destination.stat().st_size == audio.size
    ):
        return "skipped"

    # If the API omitted size, an existing file is still presumed complete
    # unless --force was requested.
    if not force and destination.exists() and audio.size is None:
        return "skipped"

    part = destination.with_name(destination.name + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            request = Request(
                audio.url,
                headers={"User-Agent": USER_AGENT},
            )
            with urlopen(request, timeout=60) as response, part.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)

            actual_size = part.stat().st_size
            if audio.size is not None and actual_size != audio.size:
                raise IOError(
                    f"size mismatch: expected {audio.size} bytes, "
                    f"received {actual_size}"
                )

            os.replace(part, destination)
            return "downloaded"

        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass

            if attempt == retries:
                break

            wait = min(2 ** (attempt - 1), 8)
            print(
                f"Download failed for {audio.original_name!r} ({exc}); "
                f"retrying in {wait}s...",
                file=sys.stderr,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Could not download {audio.original_name!r}: {last_error}"
    )


def write_manifest(
    path: Path,
    files: list[AudioFile],
    local_names: dict[str, str],
) -> None:
    rows = sorted(
        files,
        key=lambda item: local_names[item.original_name].casefold(),
    )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "original_name",
                "local_path",
                "source_url",
                "description_url",
                "bytes",
                "mime",
                "source_timestamp",
            ],
        )
        writer.writeheader()

        for audio in rows:
            writer.writerow(
                {
                    "original_name": audio.original_name,
                    "local_path": local_names[audio.original_name],
                    "source_url": audio.url,
                    "description_url": audio.description_url,
                    "bytes": "" if audio.size is None else audio.size,
                    "mime": audio.mime,
                    "source_timestamp": audio.timestamp,
                }
            )


def main() -> int:
    args = parse_args()

    if args.delay < 0:
        print("--delay cannot be negative.", file=sys.stderr)
        return 2
    if args.retries < 1:
        print("--retries must be at least 1.", file=sys.stderr)
        return 2

    print(f"Querying {args.api}")
    files = list(enumerate_audio_files(args.api, args.retries))
    files.sort(key=lambda item: item.original_name.casefold())

    if not files:
        print("No MP3 files were returned by the MediaWiki API.", file=sys.stderr)
        return 1

    known_total = sum(item.size or 0 for item in files)
    unknown_count = sum(item.size is None for item in files)

    print(f"Found {len(files)} MP3 file(s).")
    if unknown_count:
        print(
            f"Known total size: {human_size(known_total)} "
            f"({unknown_count} file(s) have unknown size)"
        )
    else:
        print(f"Total size: {human_size(known_total)}")

    local_names = unique_local_names(files)

    if args.dry_run:
        for audio in files:
            print(
                f"{human_size(audio.size):>12}  "
                f"{local_names[audio.original_name]}"
            )
        print("\nDry run only; nothing was downloaded.")
        return 0

    args.output.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failures: list[tuple[str, str]] = []

    for index, audio in enumerate(files, start=1):
        local_name = local_names[audio.original_name]
        destination = args.output / local_name

        print(
            f"[{index:>3}/{len(files)}] "
            f"{local_name} ({human_size(audio.size)})",
            end=" ... ",
            flush=True,
        )

        try:
            status = download_file(
                audio,
                destination,
                retries=args.retries,
                force=args.force,
            )
        except RuntimeError as exc:
            print("FAILED")
            failures.append((audio.original_name, str(exc)))
        else:
            print(status)
            if status == "downloaded":
                downloaded += 1
                if args.delay:
                    time.sleep(args.delay)
            else:
                skipped += 1

    manifest_path = args.output / args.manifest
    write_manifest(manifest_path, files, local_names)

    actual_total = sum(
        path.stat().st_size
        for path in args.output.glob("*.mp3")
        if path.is_file()
    )

    print()
    print(f"Downloaded: {downloaded}")
    print(f"Skipped:    {skipped}")
    print(f"Failed:     {len(failures)}")
    print(f"Audio size: {human_size(actual_total)}")
    print(f"Manifest:   {manifest_path}")

    if failures:
        print("\nFailures:", file=sys.stderr)
        for name, error in failures:
            print(f"  - {name}: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
