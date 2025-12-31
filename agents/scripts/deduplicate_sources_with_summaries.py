"""
Deduplicate links in `original sources.txt` files under the data directory.

For each `original sources.txt`, the script builds:
- dictionary1: every link with metadata (positions, summary, duplicate list, selected flag).
- dictionary2: only the selected representatives per duplicate group.

Usage:
    python deduplicate_sources.py [--data-dir .]

Notes:
- Character positions are 0-based and end positions are inclusive for easy reference
  back into the original text.
- Duplicate grouping is based on the bare-minimum form of each URL (scheme and `www`
  removed, default ports dropped, trailing slashes trimmed).
"""

from __future__ import annotations

import argparse
import json
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
    candidate = url.strip()
    # Strip surrounding markdown emphasis characters (e.g., **url** or *url*)
    candidate = re.sub(r"^[*_]+", "", candidate)
    candidate = re.sub(r"[*_]+$", "", candidate)
    # Strip trailing punctuation that often trails links
    candidate = candidate.rstrip(").,;")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", candidate):
        candidate = f"http://{candidate}"

    parsed = urlparse(candidate)
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
    """
    Extract the summary block between `start` and `end` offsets.

    Captures text after a link until the next link start (end offset provided),
    allowing multi-paragraph summaries. Trailing/leading blank lines are trimmed.
    """
    chunk = text[start:end]
    lines = [line.rstrip() for line in chunk.splitlines()]
    # Trim leading/trailing blanks
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1].strip() == "":
        lines.pop()
    if lines:
        lines[0] = lines[0].lstrip(" )]\t")
    return "\n".join(lines).strip()


def build_dictionary1(text: str) -> dict[str, dict]:
    """Parse a single `original sources.txt` content into dictionary1."""
    dictionary1: dict[str, dict] = {}

    # First, capture markdown links ([text](url)); use the target as the original form.
    md_matches = list(LINK_PATTERN.finditer(text))
    entries: list[dict] = []
    occupied_ranges: list[tuple[int, int]] = []

    for match in md_matches:
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

    # Precompute bare forms for grouping/summary extension.
    for entry in entries:
        entry["bare"] = normalize_url(entry["original form"])

    # Populate summaries for every entry using the following text until the next link.
    for i, entry in enumerate(entries):
        j = i + 1
        while j < len(entries) and entries[j]["bare"] == entry["bare"]:
            j += 1
        next_start = entries[j]["start"] if j < len(entries) else len(text)
        entry["summary"] = extract_summary(text, entry["end"] + 1, next_start)

    for idx, entry in enumerate(entries):
        link_key = f"link{idx + 1}"
        dictionary1[link_key] = {
            "duplicate list": [],
            "original form": entry["original form"],
            "bare minimum form": entry["bare"],
            "original start position of the current link": entry["start"],
            "original end position of the current link": entry["end"],
            "accompanying RAG summary + metadata string": entry["summary"],
            "selected": 0,
        }

    return dictionary1


def mark_duplicates(dictionary1: dict[str, dict]) -> dict[str, dict]:
    """Fill duplicate lists and select a representative per bare-minimum URL."""
    groups: dict[str, list[str]] = defaultdict(list)
    for key, payload in dictionary1.items():
        groups[payload["bare minimum form"]].append(key)

    for members in groups.values():
        if len(members) == 1:
            continue
        for key in members:
            dictionary1[key]["duplicate list"] = [m for m in members if m != key]

    for members in groups.values():
        winner = max(
            members,
            key=lambda k: (
                len(dictionary1[k].get("accompanying RAG summary + metadata string", "").strip()) > 0,
                len(dictionary1[k].get("accompanying RAG summary + metadata string", "")),
            ),
        )
        for key in members:
            dictionary1[key]["selected"] = 1 if key == winner else 0

    return dictionary1


def build_dictionary2(dictionary1: dict[str, dict]) -> dict[str, dict]:
    """Return only the selected entries."""
    return {k: v for k, v in dictionary1.items() if v.get("selected") == 1}


def process_file(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    dictionary1 = build_dictionary1(text)
    dictionary1 = mark_duplicates(dictionary1)
    dictionary2 = build_dictionary2(dictionary1)
    return dictionary1, dictionary2


def save_output(path: Path, dictionary1: dict[str, dict], dictionary2: dict[str, dict]) -> None:
    output1 = path.with_name("dictionary1.json")
    output2 = path.with_name("deduplicated.json")
    output1.write_text(json.dumps(dictionary1, indent=2), encoding="utf-8")
    output2.write_text(json.dumps(dictionary2, indent=2), encoding="utf-8")


def main() -> None:
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
