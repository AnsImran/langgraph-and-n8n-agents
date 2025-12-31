"""
Fetch links listed in deduplicated.json and save them as files or web pages.

Overall goal (plain English)
- You tell the script where deduplicated.json lives.
- It reads all the links, downloads files into ./files, downloads pages into
  ./web pages, and logs what happened into fetch_manifest.json.
- It uses many threads for speed, and can use Playwright to render JavaScript
  pages if needed.

Where it fits
- Point --base-dir at any vendor folder that contains deduplicated.json.
- The script writes outputs into that same folder.
"""

from __future__ import annotations

import argparse  # read command-line flags
import json  # read/write JSON files
import mimetypes  # guess file extensions
import sys  # check platform
import time  # measure how long things take
import asyncio  # set event loop policy on Windows
from pathlib import Path  # handle file system paths
from concurrent.futures import ThreadPoolExecutor, as_completed  # run work in threads

import requests  # HTTP library for GET/HEAD

try:
    from playwright.sync_api import sync_playwright  # browser automation

    PLAYWRIGHT_AVAILABLE = True  # flag if Playwright is installed
except ImportError:
    PLAYWRIGHT_AVAILABLE = False  # if not installed, skip renders

# Use Proactor loop on Windows so Playwright subprocesses can spawn cleanly.
if sys.platform.startswith("win"):
    # On Windows, use Proactor loop so Playwright can spawn processes.
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# Network timeout in seconds for HEAD/GET/navigation.
TIMEOUT = 10
# How many concurrent HTTP workers to run.
MAX_HTTP_WORKERS = 32
# How many concurrent Playwright render workers to run.
MAX_RENDER_WORKERS = 20


def clean_url(url: str) -> str:
    """Strip surrounding emphasis chars and add https:// if missing."""
    url = url.strip()  # remove spaces at ends
    url = url.strip("*_")  # remove markdown * or _ wrappers
    if not url.startswith(("http://", "https://")):  # if no scheme present
        url = "https://" + url  # assume https
    return url  # cleaned URL
    url = url.strip().strip("*_")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def classify_via_headers(url: str) -> tuple[str, str]:
    """
    Return (content-type, content-disposition) from a HEAD request.
    Falls back to empty strings on failure (some servers block HEAD).
    """
    try:
        r = requests.head(url, allow_redirects=True, timeout=TIMEOUT, headers={"User-Agent": UA})  # send HEAD
        return r.headers.get("Content-Type", "").lower(), r.headers.get("Content-Disposition", "")  # pull headers
    except Exception:
        return "", ""  # on failure, return blanks


def guess_name(url: str, disp: str, ctype: str) -> str:
    """
    Derive a filename using, in order:
    - Content-Disposition filename
    - URL path basename
    - A default based on content type
    """
    if "filename=" in disp:  # use filename from headers if present
        name = disp.split("filename=")[-1].strip("\"'; ")
        if name:
            return name
    path_part = url.split("?")[0].rstrip("/").split("/")[-1]  # last part of URL path
    if path_part:
        return path_part
    ext = mimetypes.guess_extension(ctype.split(";")[0].strip()) if ctype else ".bin"  # guess extension
    return f"download{ext or '.bin'}"  # fallback name


def save_file(url: str, ctype: str, disp: str, files_dir: Path) -> str:
    """
    Stream-download a file to files_dir and return the saved path.
    Uses content type/disp to name the file where possible.
    """
    name = guess_name(url, disp, ctype or "")  # pick a filename
    target = files_dir / name  # full path to save
    with requests.get(url, stream=True, allow_redirects=True, timeout=TIMEOUT, headers={"User-Agent": UA}) as r:
        r.raise_for_status()  # error on bad status
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):  # stream in chunks
                if chunk:
                    f.write(chunk)  # write chunk to disk
    return str(target)  # return saved path


def save_html_raw(url: str, pages_dir: Path) -> str:
    """Download raw HTML via requests and return the saved path."""
    r = requests.get(url, allow_redirects=True, timeout=TIMEOUT, headers={"User-Agent": UA})  # GET the page
    r.raise_for_status()  # error on bad status
    name = guess_name(url, "", "text/html")  # pick a filename
    if not name.lower().endswith(".html"):
        name += ".html"  # ensure .html extension
    target = pages_dir / name  # path to save
    target.write_text(r.text, encoding=r.encoding or "utf-8", errors="ignore")  # write HTML text
    return str(target)  # return saved path


def check_connectivity() -> bool:
    """Quick check: can we reach the internet? Returns True/False."""
    try:
        requests.get("https://www.google.com", timeout=3)  # tiny probe
        return True
    except Exception:
        return False


def save_html_rendered(url: str, pages_dir: Path) -> str | None:
    """
    Render the page via Playwright (if available) and return the saved path.
    Scrolls to bottom and waits briefly to trigger lazy-load content.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return None
    with sync_playwright() as p:  # start Playwright
        browser = p.chromium.launch(headless=True)  # launch headless Chromium
        page = browser.new_page()  # open a new tab
        page.goto(url, wait_until="networkidle", timeout=TIMEOUT * 1000)  # navigate and wait
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")  # scroll to bottom
        page.wait_for_timeout(2000)  # short wait for lazy content
        html = page.content()  # grab rendered HTML
        name = guess_name(url, "", "text/html")  # filename
        if not name.lower().endswith(".html"):
            name += ".rendered.html"  # mark as rendered
        target = pages_dir / name  # path to save
        target.write_text(html, encoding="utf-8", errors="ignore")  # write HTML
        browser.close()  # close browser
        return str(target)  # return saved path


def process_url_http(url: str, files_dir: Path, pages_dir: Path, pending_render: list) -> dict:
    """
    HTTP phase for a single URL:
    - HEAD to classify file vs html.
    - GET and save as file or raw html.
    - On failure, queue for Playwright render if available.
    """
    record = {"url": url, "status": "unknown", "saved": None, "type": None, "error": None}  # tracking info
    try:
        ctype, disp = classify_via_headers(url)  # try HEAD for type
        is_file = any(s in ctype for s in ["application/", "image/", "audio/", "video/"]) or "filename=" in disp
        if is_file:  # treat as file
            record["saved"] = save_file(url, ctype, disp, files_dir)
            record["type"] = "file"
            record["status"] = "ok"
            return record
        # else treat as HTML
        record["saved"] = save_html_raw(url, pages_dir)
        record["type"] = "html_raw"
        record["status"] = "ok"
        return record
    except Exception as e:
        record["error"] = repr(e)  # note error
        if PLAYWRIGHT_AVAILABLE:
            record["status"] = "pending_render"  # flag for render fallback
            pending_render.append(record)  # queue for later
        else:
            record["status"] = "failed"  # no render available
        return record  # return failure/pending


def main(base_dir: Path) -> None:
    """
    Orchestrate the full fetch:
    - Load deduplicated.json.
    - Parallel HTTP fetches (files/raw HTML).
    - Parallel Playwright renders for pending URLs.
    - Write manifest.
    """
    try:
        # Check basic internet connectivity before doing heavy work.
        if not check_connectivity():
            print("Warning: Internet connectivity check failed (google.com unreachable). Continuing anyway...")

        dedup_file = base_dir / "deduplicated.json"  # path to dedup file
        if not dedup_file.exists():
            raise SystemExit(f"deduplicated.json not found at {dedup_file}")  # fail if missing

        files_dir = base_dir / "files"  # output dir for files
        pages_dir = base_dir / "web pages"  # output dir for pages
        files_dir.mkdir(exist_ok=True)  # create if missing
        pages_dir.mkdir(exist_ok=True)

        data = json.loads(dedup_file.read_text(encoding="utf-8"))  # load JSON
        urls = []  # list of URLs to fetch
        for key, entry in data.items():  # loop entries
            if key == "_meta":  # skip meta
                continue
            raw = entry.get("original form") or entry.get("bare minimum form") or ""  # pick URL
            if raw:
                urls.append(clean_url(raw))  # clean and store

        manifest: list[dict] = []  # results list
        pending_render: list[dict] = []  # queue for Playwright

        start = time.time()  # start timer

        # Phase 1: HTTP in parallel
    with ThreadPoolExecutor(max_workers=MAX_HTTP_WORKERS) as pool:  # HTTP pool
        futures = {pool.submit(process_url_http, u, files_dir, pages_dir, pending_render): u for u in urls}  # submit all
        for fut in as_completed(futures):  # as each finishes
            try:
                rec = fut.result()  # get record
            except Exception as e:
                rec = {
                    "url": "<unknown>",
                    "status": "failed",
                    "type": None,
                    "saved": None,
                    "error": f"HTTP worker error: {repr(e)}",
                }
            manifest.append(rec)  # store
            print(f"{rec['status']:12} {rec.get('type') or '-':12} {rec.get('url')} -> {rec.get('saved')}")  # log

        # Phase 2: Render pending URLs (concurrent)
        if PLAYWRIGHT_AVAILABLE and pending_render:
            print(f"\nRendering {len(pending_render)} URLs via Playwright (up to {MAX_RENDER_WORKERS} at a time)...")

            def render_one(rec: dict) -> dict:
                """Helper to render one pending URL."""
                try:
                    saved = save_html_rendered(rec["url"], pages_dir)  # try render
                    rec["saved"] = saved
                    rec["type"] = "html_rendered"
                    rec["status"] = "ok"
                    rec["error"] = None
                except Exception as e:
                    rec["status"] = "failed"
                    rec["error"] = repr(e)
                return rec  # return updated record

            with ThreadPoolExecutor(max_workers=MAX_RENDER_WORKERS) as pool:  # Playwright pool
                futures = {pool.submit(render_one, rec): rec for rec in pending_render}  # submit renders
                for fut in as_completed(futures):  # as each finishes
                    try:
                        rec = fut.result()
                    except Exception as e:
                        rec = {
                            "url": "<unknown>",
                            "status": "failed",
                            "type": None,
                            "saved": None,
                            "error": f"Render worker error: {repr(e)}",
                        }
                    if rec["status"] == "ok":
                        print(f"ok           html_rendered  {rec['url']} -> {rec['saved']}")  # success log
                    else:
                        print(f"failed       -              {rec.get('url')} -> {rec.get('error')}")  # error log

        manifest_path = base_dir / "fetch_manifest.json"  # manifest path
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")  # save manifest
        print(f"\nDone in {time.time() - start:0.2f}s. Manifest: {manifest_path}")  # final log
    except Exception as e:
        # Catch any unexpected top-level error and print it clearly.
        print(f"Fatal error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch links from deduplicated.json")  # CLI parser
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Directory containing deduplicated.json (default: current directory)",
    )
    args = parser.parse_args()  # parse args
    main(args.base_dir)  # run main with provided base dir
