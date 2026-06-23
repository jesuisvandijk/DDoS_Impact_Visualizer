"""
scrape_and_build_dataset.py
────────────────────────────
Reads an Excel file with a Link column, scrapes full article text from each
URL (stripping ads, nav bars, footers, cookie banners via trafilatura),
then writes a column-oriented JSON ready for regressor_jesse.py and
article_to_event_level.py.

Usage:
  python scrape_and_build_dataset.py --input alerts.xlsx --output new_dataset.json

Optional flags:
  --sheet         Sheet name or 0-based index (default: 0)
  --workers       Parallel download threads (default: 6)
  --timeout       Hard per-URL socket timeout in seconds (default: 12)
  --delay         Polite delay between requests per thread (default: 0.5)
  --fallback-col  Excel column used when scraping fails (default: Content)
  --type-filter   Keep only rows where Type == value (e.g. News)
  --drop-failed   Drop rows where scraping failed (default: keep with fallback)
  --max-rows      Only process first N rows — useful for testing
  --cache         JSON cache file path (default: scrape_cache.json)
"""

import argparse
import json
import sys
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path

import requests
import pandas as pd
import trafilatura
from langdetect import detect, LangDetectException


# ── Constants ─────────────────────────────────────────────────────────────────

COLUMN_MAP = {
    "Type":    "Alert Type",
    "Content": "_content_preview",   # kept as fallback only
    "Date":    "Date",
    "Link":    "Link",
    "Title":   "Content_Title",
}

REQUIRED_OUTPUT_COLS = ["Alert Type", "Content", "Date", "Link"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_cache_lock = threading.Lock()


# ── Scraper ───────────────────────────────────────────────────────────────────

def scrape_url(url: str, timeout: int) -> str | None:
    """
    Fetch with requests (guaranteed hard socket timeout — no stalling),
    then extract clean article text with trafilatura (removes ads, nav,
    footers, sidebars, cookie banners automatically).
    Returns None on any failure so the caller falls back gracefully.
    """
    if not isinstance(url, str) or not url.startswith("http"):
        return None
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        html = resp.text
    except Exception:
        return None

    try:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_precision=True,
        )
        return text.strip() if text else None
    except Exception:
        return None


# ── Cache ─────────────────────────────────────────────────────────────────────

def load_cache(cache_path: str) -> dict:
    p = Path(cache_path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Loaded {len(data):,} cached URLs from {cache_path}")
        return data
    return {}


def save_cache(cache: dict, cache_path: str) -> None:
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


# ── Parallel fetching ─────────────────────────────────────────────────────────

def scrape_all(
    urls: list,
    timeout: int,
    workers: int,
    delay: float,
    cache: dict,
    cache_path: str,
) -> dict:
    results = {}
    to_fetch = []

    for url in urls:
        if url in cache:
            results[url] = cache[url]
        else:
            to_fetch.append(url)

    print(f"  {len(results):,} from cache  |  {len(to_fetch):,} to fetch")

    if not to_fetch:
        return results

    completed = 0
    total = len(to_fetch)
    save_every = 25

    def fetch_one(url):
        time.sleep(random.uniform(delay * 0.5, delay * 1.5))
        return url, scrape_url(url, timeout)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, url): url for url in to_fetch}
        for future in as_completed(futures):
            url, text = future.result()
            results[url] = text
            with _cache_lock:
                cache[url] = text
            completed += 1
            status = "OK" if text else "--"
            # Print every URL so you can see progress and spot stalls
            print(f"  [{completed:>4}/{total}] {status}  {url[:90]}")
            if completed % save_every == 0:
                with _cache_lock:
                    save_cache(cache, cache_path)

    save_cache(cache, cache_path)
    return results


# ── Excel loading ─────────────────────────────────────────────────────────────

def load_excel(path: str, sheet) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, sheet_name=sheet, dtype=str)
    except FileNotFoundError:
        sys.exit(f"[ERROR] File not found: {path}")
    except Exception as e:
        sys.exit(f"[ERROR] Could not read Excel: {e}")
    df.columns = [c.strip() for c in df.columns]
    print(f"  Loaded {len(df):,} rows  |  columns: {list(df.columns)}")
    return df


def remap(df: pd.DataFrame) -> pd.DataFrame:
    present = {src: dst for src, dst in COLUMN_MAP.items() if src in df.columns}
    missing = [src for src in COLUMN_MAP if src not in df.columns]
    if missing:
        print(f"  [WARNING] Excel columns not found (skipped): {missing}")
    return df[list(present.keys())].rename(columns=present).copy()


# ── Assemble Content column ───────────────────────────────────────────────────

def apply_scrape_results(df, scrape_results, fallback_col, drop_failed):
    def pick(row):
        text = scrape_results.get(str(row.get("Link", "")), None)
        if text:
            return text
        return str(row.get(fallback_col, "") or "")

    df["Content"] = df.apply(pick, axis=1)
    df["scrape_ok"] = df["Link"].map(lambda u: bool(scrape_results.get(str(u))))

    n_ok   = int(df["scrape_ok"].sum())
    n_fail = int((~df["scrape_ok"]).sum())
    print(f"\n  Scraped OK  : {n_ok:,}")
    print(f"  Fallback    : {n_fail:,}  (paywalled / dead / blocked)")

    if drop_failed:
        df = df[df["scrape_ok"]].reset_index(drop=True)
        print(f"  Rows after dropping failed: {len(df):,}")

    df = df.drop(columns=["scrape_ok", "_content_preview"], errors="ignore")
    return df


# ── JSON output ───────────────────────────────────────────────────────────────

def to_column_oriented_json(df, output_path):
    df = df.reset_index(drop=True)
    df.index = df.index.astype(str)
    out = {col: df[col].to_dict() for col in df.columns}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Written {len(df):,} articles → {output_path}")
    print(f"  Columns: {list(df.columns)}")


def preview(df, n=3):
    print(f"\n── Preview (first {n} rows) " + "─" * 50)
    for i, row in df.head(n).iterrows():
        body = str(row.get("Content", ""))
        title = str(row.get("Content_Title", row.get("Link", "")))[:80]
        print(f"  [{i}] {title}")
        print(f"       chars : {len(body)}")
        print(f"       text  : {body[:120]}{'…' if len(body) > 120 else ''}\n")


# ── Language filtering ─────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Returns an ISO 639-1 language code (e.g. 'en', 'fr', 'de') or 'unknown'
    if detection fails (e.g. text too short or empty).
    """
    text = (text or "").strip()
    if len(text) < 20:          # too short to detect reliably
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def filter_english(df: pd.DataFrame, drop_non_english: bool) -> pd.DataFrame:
    """
    Detects language of the (post-scrape) Content column and either drops
    non-English rows or just flags them with a 'language' column.
    """
    print("\n  Detecting language of scraped content…")
    df["language"] = df["Content"].map(detect_language)

    counts = df["language"].value_counts()
    print("  Language breakdown:")
    for lang, n in counts.items():
        print(f"    {lang:8s} : {n:,}")

    if drop_non_english:
        before = len(df)
        df = df[df["language"] == "en"].reset_index(drop=True)
        print(f"  Dropped non-English rows: {len(df):,} / {before:,} kept")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        required=True)
    parser.add_argument("--output",       required=True)
    parser.add_argument("--sheet",        default=0)
    parser.add_argument("--workers",      type=int,   default=6)
    parser.add_argument("--timeout",      type=int,   default=12)
    parser.add_argument("--delay",        type=float, default=0.5)
    parser.add_argument("--fallback-col", default="Content")
    parser.add_argument("--type-filter",  default=None)
    parser.add_argument("--type-match",   choices=["exact", "contains"], default="contains",
                        help="'contains' (default) matches substring case-insensitively; "
                             "'exact' requires exact match after lowering/stripping")
    parser.add_argument("--english-only", action="store_true",
                        help="Detect language of scraped content and drop non-English rows")
    parser.add_argument("--drop-failed",  action="store_true")
    parser.add_argument("--max-rows",     type=int, default=None,
                        help="Only process first N rows (for testing)")
    parser.add_argument("--cache",        default="scrape_cache.json")
    parser.add_argument("--target-count", type=int, default=None,
                        help="Trim final output to exactly N rows (after all filtering). "
                             "Warns if fewer than N rows survive filtering.")
    args = parser.parse_args()

    try:
        sheet = int(args.sheet)
    except ValueError:
        sheet = args.sheet

    print("\nStep 1: Loading Excel…")
    df_raw = load_excel(args.input, sheet)

    print("\nStep 2: Remapping columns…")
    df = remap(df_raw)

    if args.type_filter and "Alert Type" in df.columns:
        before = len(df)
        print(f"  Distinct 'Alert Type' values found: {sorted(df['Alert Type'].dropna().unique().tolist())}")
        target = args.type_filter.strip().lower()
        mask = df["Alert Type"].fillna("").str.strip().str.lower()
        if args.type_match == "exact":
            df = df[mask == target].reset_index(drop=True)
        else:  # contains
            df = df[mask.str.contains(target, na=False)].reset_index(drop=True)
        print(f"  Type filter '{args.type_filter}' ({args.type_match}): {len(df):,} / {before:,} rows kept")

    if args.max_rows:
        df = df.head(args.max_rows).reset_index(drop=True)
        print(f"  Subset: using first {len(df):,} rows (--max-rows {args.max_rows})")

    print("\nStep 3: Loading URL cache…")
    cache = load_cache(args.cache)

    print(f"\nStep 4: Scraping ({args.workers} workers, {args.timeout}s hard timeout per URL)…")
    urls = df["Link"].fillna("").astype(str).tolist()

    scrape_results = scrape_all(
        urls,
        timeout=args.timeout,
        workers=args.workers,
        delay=args.delay,
        cache=cache,
        cache_path=args.cache,
    )

    print("\nStep 5: Assembling Content column…")
    fallback_col = "_content_preview" if args.fallback_col == "Content" else args.fallback_col
    df = apply_scrape_results(df, scrape_results, fallback_col, args.drop_failed)

    if args.english_only:
        print("\nStep 5b: Filtering to English-language content…")
        df = filter_english(df, drop_non_english=True)

    if args.target_count:
        if len(df) < args.target_count:
            print(f"\n[WARNING] Only {len(df):,} rows survived filtering — "
                  f"fewer than the requested --target-count {args.target_count:,}.")
            print("  Options: rerun without --max-rows to scrape more candidates from the Excel file,")
            print("  or lower --target-count to match what's actually available.")
        else:
            df = df.head(args.target_count).reset_index(drop=True)
            print(f"\n  Trimmed to exactly {len(df):,} rows (--target-count {args.target_count:,})")

    preview(df)

    missing = [c for c in REQUIRED_OUTPUT_COLS if c not in df.columns]
    if missing:
        print(f"[WARNING] Missing required pipeline columns: {missing}")

    df_out = df.drop(columns=["language"], errors="ignore")

    print("Step 6: Writing JSON…")
    to_column_oriented_json(df_out, args.output)

    print(f"\nDone!")
    print(f"  Set DATA_FILE = '{args.output}' in regressor_jesse.py and article_to_event_level.py")
    print(f"  Cache saved to {args.cache} — re-runs skip already-fetched URLs")

if __name__ == "__main__":
    main()

