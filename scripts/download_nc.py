#!/usr/bin/env python3
"""
interactive_download.py

Interactive, user-friendly Argo downloader.

Features:
 - Interactive prompts asking what to target (manifest vs data).
 - Pre-defined shorthand for GEO and DAC base URLs; also accepts custom base URL.
 - Ask for year/month/ocean when relevant.
 - Shows a preview and count of files found and asks for confirmation before downloading.
 - Still supports CLI args for automation.
 - Saves an auto_manifest for reproducibility and logs failures.

Usage (interactive):
    python interactive_download.py

Usage (non-interactive / scripted):
    python interactive_download.py --source geo --ocean indian_ocean --year 2022 --month 01 --mode data --outdir downloads

Dependencies:
    pip install requests beautifulsoup4 tqdm pandas
"""
import os
import sys
import time
import csv
import argparse
import requests
import pandas as pd
from urllib.parse import urljoin
from tqdm import tqdm
from bs4 import BeautifulSoup

# ---------------- Config ----------------
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 4
SLEEP_BETWEEN_RETRIES = 2.0
CHUNK_SIZE = 8192

PRESET_BASES = {
    "geo": "https://data-argo.ifremer.fr/geo/",
    "dac": "https://data-argo.ifremer.fr/dac/",
    # Add more presets here if you use other mirrors
}

# ----------------------------------------

def safe_join_url(base, rel):
    return urljoin(base, rel.lstrip("/"))

def ensure_dir_for(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def list_files_from_url(base_url, exts=(".nc", ".parquet"), timeout=DEFAULT_TIMEOUT):
    """
    Parse an HTML directory listing and return relative file paths (href values)
    that end with one of the extensions in exts.
    """
    try:
        r = requests.get(base_url, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Could not reach {base_url}: {e}")
    if r.status_code != 200:
        raise RuntimeError(f"Could not list {base_url} (status {r.status_code})")

    soup = BeautifulSoup(r.text, "html.parser")
    files = []
    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue
        for ext in exts:
            if href.lower().endswith(ext):
                files.append(href)
                break
    # Remove duplicates while preserving order
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq

def download_one(url, out_path, timeout=DEFAULT_TIMEOUT):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True, "exists"

    ensure_dir_for(out_path)
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            stream_resp = requests.get(url, stream=True, timeout=timeout)
            if stream_resp.status_code != 200:
                last_exc = f"status {stream_resp.status_code}"
                time.sleep(SLEEP_BETWEEN_RETRIES)
                continue

            with open(out_path + ".part", "wb") as fh:
                for chunk in stream_resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
            os.replace(out_path + ".part", out_path)
            return True, "downloaded"
        except Exception as e:
            last_exc = str(e)
            time.sleep(SLEEP_BETWEEN_RETRIES)
    return False, last_exc or "failed"

def prompt_choice(prompt, choices, default=None):
    """
    Prompt user to pick one of choices (case-insensitive).
    choices: list of (key, description) or list of strings.
    Returns the chosen key/string.
    """
    if isinstance(choices[0], tuple):
        mapping = {k.lower(): (k, desc) for (k, desc) in choices}
        prompt_lines = [f"  {k}: {desc}" for (k, desc) in choices]
        prompt_text = prompt + "\n" + "\n".join(prompt_lines) + "\nChoice"
    else:
        mapping = {c.lower(): c for c in choices}
        prompt_text = prompt + f" ({'/'.join(choices)})"

    if default:
        prompt_text += f" [default: {default}]"
    prompt_text += ": "

    while True:
        ans = input(prompt_text).strip()
        if ans == "" and default is not None:
            ans = default
        if ans.lower() in mapping:
            return mapping[ans.lower()][0] if isinstance(choices[0], tuple) else mapping[ans.lower()]
        print("Invalid choice. Please try again.")

def prompt_text(prompt, default=None, allow_empty=False):
    t = prompt
    if default:
        t += f" [default: {default}]"
    t += ": "
    while True:
        v = input(t).strip()
        if v == "" and default is not None:
            return default
        if v == "" and allow_empty:
            return ""
        if v != "":
            return v
        print("Input cannot be empty.")

def validate_year(y):
    try:
        iv = int(y)
        if 1900 <= iv <= 2100:
            return True
    except:
        pass
    return False

def validate_month(m):
    try:
        iv = int(m)
        return 1 <= iv <= 12
    except:
        return False

def build_base_url(interactive_vals):
    """
    interactive_vals: dict with keys:
      - source (geo/dac/custom)
      - base_url (if custom or provided)
      - ocean (for geo) [string, optional]
      - year (optional)
      - month (optional)
    """
    source = interactive_vals.get("source")
    base = interactive_vals.get("base_url")
    ocean = interactive_vals.get("ocean")
    year = interactive_vals.get("year")
    month = interactive_vals.get("month")

    if source == "custom":
        base_url = base
    else:
        base_url = PRESET_BASES.get(source)
        if base_url is None:
            raise RuntimeError(f"No preset for source '{source}'")

        if source == "geo":
            # append ocean if provided (support both 'indian_ocean' or '' for global)
            if ocean:
                base_url = safe_join_url(base_url, ocean.rstrip("/") + "/")
        # For DAC we keep top-level preset and then append year/month if provided
    if year:
        base_url = safe_join_url(base_url, str(year).zfill(4) + "/")
    if month:
        base_url = safe_join_url(base_url, str(month).zfill(2) + "/")
    return base_url

def interactive_flow(args):
    """
    Gather interactive options from the user (if CLI did not supply them).
    Returns a dict of final parameters to run.
    """
    params = {}

    # 1) Mode: manifest or data
    mode = args.mode or prompt_choice("Which target do you want to download?", ["data", "manifest"], default="data")
    params["mode"] = mode

    # 2) Source selection (geo, dac, custom)
    source = args.source
    if not source:
        source = prompt_choice(
            "Select data source / structure",
            [("geo", "GEO organization (ocean/year/month) — easy for month-by-month pulls"),
             ("dac", "DAC organization (raw per-float archives) — source-of-truth"),
             ("custom", "Custom base URL (you supply full URL)")]
        )
    params["source"] = source

    # If custom and not provided base_url via args, ask
    if source == "custom":
        base_url = args.base_url or prompt_text("Enter custom base URL (must include protocol, e.g. https://.../)", default="")
        if not base_url.endswith("/"):
            base_url += "/"
        params["base_url"] = base_url
    else:
        params["base_url"] = args.base_url or PRESET_BASES.get(source)

    # For GEO optionally ask ocean
    ocean = args.ocean
    if source == "geo" and not ocean:
        ocean = prompt_text("Enter ocean folder for GEO (e.g. indian_ocean) or leave empty for top-level", default="", allow_empty=True)
    params["ocean"] = ocean

    # Year/month: ask only if mode == data (manifests are usually top-level)
    year = args.year
    month = args.month
    if params["mode"] == "data":
        if not year:
            year = prompt_text("Enter year (YYYY) or leave empty to skip", default="", allow_empty=True)
        if year and not validate_year(year):
            print("Invalid year. It must be a number like 2020. Aborting.")
            sys.exit(1)
        if not month and year:
            month = prompt_text("Enter month (1-12) or leave empty to skip", default="", allow_empty=True)
        if month and not validate_month(month):
            print("Invalid month. Use 1-12 (or 01-12). Aborting.")
            sys.exit(1)
    else:
        # manifest mode: allow user to download manifests like ar_index_global_prof.txt.gz
        # we will filter for .txt or .txt.gz by default for manifests
        year = year or ""
        month = month or ""

    params["year"] = year
    params["month"] = month

    # Outdir
    outdir = args.outdir or prompt_text("Output directory", default="argo_downloads")
    params["outdir"] = outdir

    # File extension filter
    if params["mode"] == "manifest":
        exts = (".txt", ".txt.gz")
    else:
        # data mode: allow both .nc and .parquet; user can change if desired
        exts = (".nc", ".parquet")
    params["exts"] = exts

    # Quick final summary
    print("\nSummary of choices:")
    print(f"  Mode      : {params['mode']}")
    print(f"  Source    : {params['source']}")
    print(f"  Base URL  : {params['base_url']}")
    if params['ocean']:
        print(f"  Ocean     : {params['ocean']}")
    if params['year']:
        print(f"  Year      : {params['year']}")
    if params['month']:
        print(f"  Month     : {params['month']}")
    print(f"  Extensions: {params['exts']}")
    print(f"  Out dir   : {params['outdir']}\n")

    return params

def run_process(params, yes_all=False):
    # Build final base URL
    base_url = build_base_url(params)

    print(f"Listing files at: {base_url}")
    try:
        files = list_files_from_url(base_url, exts=params["exts"])
    except Exception as e:
        print(f"Error listing files: {e}", file=sys.stderr)
        return

    if not files:
        print("No files found at that location with the requested extensions.")
        return

    print(f"Found {len(files)} file(s). Sample:")
    for i, f in enumerate(files[:10]):
        print(f"  {i+1}. {f}")
    if len(files) > 10:
        print("  ...")

    # Save manifest for reproducibility
    manifest_path = os.path.join(params["outdir"], "auto_manifest.csv")
    ensure_dir_for(manifest_path)
    pd.DataFrame({"file": files}).to_csv(manifest_path, index=False)
    print(f"\nManifest saved to: {manifest_path}")

    if not yes_all:
        ans = input("Proceed to download all found files? (y/N): ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted by user.")
            return

    ok = skipped = failed = 0
    failures = []
    for rel_path in tqdm(files, desc="Downloading", unit="file"):
        url = safe_join_url(base_url, rel_path)
        out_path = os.path.join(params["outdir"], rel_path.replace("/", os.sep))
        succ, reason = download_one(url, out_path)
        if succ:
            if reason == "exists":
                skipped += 1
            else:
                ok += 1
        else:
            failed += 1
            failures.append((rel_path, reason))

    print("\nDownload summary:")
    print(f"  Downloaded : {ok}")
    print(f"  Skipped    : {skipped}")
    print(f"  Failed     : {failed}")

    if failures:
        fail_csv = os.path.join(params["outdir"], "failed_downloads.csv")
        ensure_dir_for(fail_csv)
        with open(fail_csv, "w", newline="", encoding="utf8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "reason"])
            w.writerows(failures)
        print(f"Failures logged to: {fail_csv}")

def main():
    parser = argparse.ArgumentParser(description="Interactive Argo downloader.")
    parser.add_argument("--manifest", help="Path to manifest CSV (with 'file' column). If provided, script will download files listed in it.")
    parser.add_argument("--source", choices=["geo", "dac", "custom"], help="Preset source to use instead of interactive prompt.")
    parser.add_argument("--base-url", help="If using custom base URL (or to override presets). Must end with '/'.")
    parser.add_argument("--ocean", help="For GEO: ocean folder name, e.g. indian_ocean.")
    parser.add_argument("--year", help="Year (YYYY).")
    parser.add_argument("--month", help="Month (01-12 or 1-12).")
    parser.add_argument("--mode", choices=["data", "manifest"], help="Whether to download regular data files or manifest/index files.")
    parser.add_argument("--outdir", help="Output directory.")
    parser.add_argument("--yes", action="store_true", help="Assume yes for confirmations (non-interactive).")
    args = parser.parse_args()

    if args.manifest:
        # If manifest provided, do non-interactive manifest-download flow
        df = pd.read_csv(args.manifest, dtype=str)
        if "file" not in df.columns:
            print("Manifest must contain a 'file' column", file=sys.stderr)
            sys.exit(1)
        files = df["file"].dropna().tolist()
        base_url = args.base_url or "https://data-argo.ifremer.fr/"
        params = {
            "mode": "data",
            "source": args.source or "custom",
            "base_url": base_url,
            "ocean": args.ocean or "",
            "year": "",
            "month": "",
            "exts": (".nc", ".parquet"),
            "outdir": args.outdir or "argo_downloads"
        }

        print(f"Will download {len(files)} items listed in manifest using base URL {base_url}.")
        if not args.yes:
            ok = input("Proceed? (y/N): ").strip().lower()
            if ok not in ("y", "yes"):
                print("Aborted.")
                return

        # Download manifest-listed files directly
        ensure_dir_for(params["outdir"])
        ok = skipped = failed = 0
        failures = []
        for rel_path in tqdm(files, desc="Downloading", unit="file"):
            url = safe_join_url(base_url, rel_path)
            out_path = os.path.join(params["outdir"], rel_path.replace("/", os.sep))
            succ, reason = download_one(url, out_path)
            if succ:
                if reason == "exists":
                    skipped += 1
                else:
                    ok += 1
            else:
                failed += 1
                failures.append((rel_path, reason))

        print("\nSummary:")
        print(f"Downloaded: {ok}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")
        if failures:
            fail_csv = os.path.join(params["outdir"], "failed_downloads.csv")
            ensure_dir_for(fail_csv)
            with open(fail_csv, "w", newline="", encoding="utf8") as fh:
                w = csv.writer(fh)
                w.writerow(["file", "reason"])
                w.writerows(failures)
            print(f"Failures logged to: {fail_csv}")

    else:
        # Interactive mode to build the target
        params = interactive_flow(args)
        run_process(params, yes_all=args.yes)

if __name__ == "__main__":
    main()
