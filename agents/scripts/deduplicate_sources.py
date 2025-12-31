"""
Overview
--------
This script scans vendor subfolders that contain an `original sources.txt`, extracts
all links in those files, and produces:
  - dictionary1.json: every link with metadata such as positions, normalized form,
    and duplicate lists.
  - deduplicated.json: only one representative per normalized (bare) URL.

Goals and fit
-------------
- Normalize and de-duplicate links coming from many LLM outputs with inconsistent
  formatting (markdown links, raw URLs, emphasized links, etc.).
- Provide deterministic selection of one link per duplicate group (first occurrence).
- Leave summaries empty in this variant; a separate script handles summary-aware
  selection.

Usage
-----
    python deduplicate_sources.py [--data-dir .]

Key behaviors
-------------
- Character positions are 0-based; end positions are inclusive.
- Duplicate grouping is driven by a normalized "bare minimum" URL
  (scheme/`www` stripped, default ports dropped, tracking params removed, aliases
  applied).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

LINK_PATTERN = re.compile(r"\[(?P<link_text>[^\]]+)\]\((?P<link_target>[^)]+)\)")
# Raw URL pattern captures:
# - explicit schemes (http/https)
# - www.* domains
# - bare domains with a path (e.g., example.com/foo)
RAW_URL_PATTERN = re.compile(
    r"""
    (?P<url>
        (?:https?://|www\.)[^\s)]+
        |
        [a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s)]*)?
    )
    """,
    re.VERBOSE,
)


def normalize_url(url: str) -> str:
    """Return a bare-minimum form of the URL for duplicate detection."""
    # Clean markdown emphasis and trailing punctuation.
    candidate = url.strip()
    # Strip surrounding markdown emphasis characters (e.g., **url** or *url*)
    candidate = re.sub(r"^[*_]+", "", candidate)
    candidate = re.sub(r"[*_]+$", "", candidate)
    # Strip trailing punctuation that often trails links
    candidate = candidate.rstrip(").,;")
    # Ensure a scheme exists for urlparse.
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", candidate):
        candidate = f"http://{candidate}"

    # Parse and normalize host/path/query.
    try:
        parsed = urlparse(candidate)
    except ValueError:
        # Handle malformed URLs (e.g., stray brackets) by stripping brackets and retrying.
        candidate_fallback = candidate.strip("[]")
        try:
            parsed = urlparse(candidate_fallback)
        except ValueError:
            # If still invalid, return the cleaned candidate as-is to avoid crashes.
            return candidate_fallback
    netloc = parsed.netloc.lower()

    if netloc.startswith("www."):
        netloc = netloc[4:]

    # Host aliases (add more as needed)
    host_aliases = {
        "agfa.com": "agfahealthcare.com",
        "www.agfa.com": "agfahealthcare.com",
    }
    netloc = host_aliases.get(netloc, netloc)

    if netloc.endswith(":80"):
        netloc = netloc[: -len(":80")]
    if netloc.endswith(":443"):
        netloc = netloc[: -len(":443")]

    # Normalize path (collapse duplicate slashes, drop trailing slash).
    path = re.sub(r"/{2,}", "/", parsed.path or "").rstrip("/")

    # Strip tracking params and fragments
    tracking_prefixes = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid", "vero_id")
    query_pairs = []
    if parsed.query:
        for key, value in [q.split("=", 1) if "=" in q else (q, "") for q in parsed.query.split("&") if q]:
            if key.startswith(tracking_prefixes):
                continue
            query_pairs.append((key, value))
    query = "&".join(f"{k}={v}" if v else k for k, v in query_pairs)

    normalized = netloc + path
    if query:
        normalized = f"{normalized}?{query}"

    return normalized


def extract_summary(text: str, start: int, end: int) -> str:
    """Placeholder: summaries are intentionally not extracted in this variant."""
    return ""


def build_dictionary1(text: str) -> dict[str, dict]:
    """Parse one `original sources.txt` into dictionary1 entries with metadata."""
    dictionary1: dict[str, dict] = {}

    # First, capture markdown links ([text](url)); use the target as the original form.
    md_matches = list(LINK_PATTERN.finditer(text))
    entries: list[dict] = []
    occupied_ranges: list[tuple[int, int]] = []

    for match in md_matches:
        # Store the link target and mark both target and link-text spans as occupied.
        original_form = match.group("link_target").strip()
        start_pos = match.start("link_target")
        end_pos = match.end("link_target") - 1  # inclusive
        occupied_ranges.append((start_pos, end_pos))
        occupied_ranges.append((match.start("link_text"), match.end("link_text") - 1))
        entries.append(
            {
                "original form": original_form,
                "start": start_pos,
                "end": end_pos,
            }
        )

    # Next, capture raw URLs not already covered by markdown link targets.
    def overlaps(span: tuple[int, int]) -> bool:
        return any(not (span[1] < s or span[0] > e) for s, e in occupied_ranges)

    for match in RAW_URL_PATTERN.finditer(text):
        start_pos = match.start("url")
        end_pos = match.end("url") - 1
        if overlaps((start_pos, end_pos)):
            continue
        # Record raw URL and mark its span.
        original_form = match.group("url").strip()
        entries.append(
            {
                "original form": original_form,
                "start": start_pos,
                "end": end_pos,
            }
        )
        occupied_ranges.append((start_pos, end_pos))

    # Sort by position to keep deterministic link numbering.
    entries.sort(key=lambda x: x["start"])

    # Precompute bare forms for grouping.
    for entry in entries:
        entry["bare"] = normalize_url(entry["original form"])

    for idx, entry in enumerate(entries):
        link_key = f"link{idx + 1}"
        dictionary1[link_key] = {
            "duplicate list": [],
            "original form": entry["original form"],
            "bare minimum form": entry["bare"],
            "original start position of the current link": entry["start"],
            "original end position of the current link": entry["end"],
            "accompanying RAG summary + metadata string": "",
            "selected": 0,
        }

    return dictionary1


def mark_duplicates(dictionary1: dict[str, dict]) -> dict[str, dict]:
    """Fill duplicate lists and select the first-occurring representative per bare URL."""
    groups: dict[str, list[str]] = defaultdict(list)
    for key, payload in dictionary1.items():
        groups[payload["bare minimum form"]].append(key)

    for members in groups.values():
        if len(members) == 1:
            continue
        # Populate duplicate lists for members in the same group.
        for key in members:
            dictionary1[key]["duplicate list"] = [m for m in members if m != key]

    for members in groups.values():
        # Select the first occurrence (deterministic, based on document order)
        winner = members[0]
        for key in members:
            dictionary1[key]["selected"] = 1 if key == winner else 0

    return dictionary1


def build_dictionary2(dictionary1: dict[str, dict]) -> dict[str, dict]:
    """Return only the selected entries."""
    return {k: v for k, v in dictionary1.items() if v.get("selected") == 1}


def process_file(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Parse one `original sources.txt`, de-duplicate links, and return both dictionaries.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    dictionary1 = build_dictionary1(text)
    dictionary1 = mark_duplicates(dictionary1)
    dictionary2 = build_dictionary2(dictionary1)
    return dictionary1, dictionary2


def save_output(path: Path, dictionary1: dict[str, dict], dictionary2: dict[str, dict]) -> None:
    """
    Write dictionary1.json and deduplicated.json next to the source file, with meta counts.
    """
    output1 = path.with_name("dictionary1.json")
    output2 = path.with_name("deduplicated.json")
    output1.write_text(json.dumps(dictionary1, indent=2), encoding="utf-8")
    output2.write_text(json.dumps(dictionary2, indent=2), encoding="utf-8")


def main() -> None:
    """Entry point: find source files under data-dir and process each."""
    parser = argparse.ArgumentParser(description="Deduplicate links inside original sources.txt files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Base directory containing subfolders with original sources.txt files (default: current directory).",
    )
    args = parser.parse_args()

    source_files = list(args.data_dir.glob("*/original sources.txt"))
    if not source_files:
        raise SystemExit("No 'original sources.txt' files found under the provided data directory.")

    for src in source_files:
        print(f"Processing {src} ...")
        dictionary1, dictionary2 = process_file(src)
        save_output(src, dictionary1, dictionary2)
        print(
            f"  links found: {len(dictionary1)} | deduplicated: {len(dictionary2)} | output: {src.with_name('deduplicated.json')}"
        )


if __name__ == "__main__":
    main()
