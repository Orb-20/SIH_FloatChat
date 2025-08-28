# scripts/nc_month_to_parquet.py
import os
import glob
import math
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/interim")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_reference_date(s):
    # ARGO REFERENCE_DATE_TIME is typically YYYYMMDDHHMISS (or shorter variants)
    if s is None:
        return None
    s = str(s).strip()
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def juld_to_datetime(juld_vals, reference_date_time, juld_units_hint=None):
    """
    Convert numeric JULD values to numpy datetime64[ns] array.
    Handles:
    - missing / fill values (very large numbers) -> NaT
    - seconds vs days units hints
    - overflow protection
    Returns: numpy array dtype datetime64[ns]
    """
    ref = reference_date_time or datetime(1950, 1, 1)
    # coerce to numeric 1D array
    try:
        arr = np.array(juld_vals, dtype=float).ravel()
    except Exception:
        arr = np.array([], dtype=float)

    if arr.size == 0:
        return np.array([], dtype="datetime64[ns]")

    out = []
    # define threshold for obviously invalid fill values (common netCDF fills are huge)
    FILL_THRESHOLD = 1e8

    units_hint = (juld_units_hint or "").lower()

    for s in arr:
        # guard NaN, inf, extremely large numbers
        if s is None or np.isnan(s) or np.isinf(s) or abs(s) > FILL_THRESHOLD:
            out.append(np.datetime64("NaT"))
            continue

        # attempt conversion with units hint first
        try:
            if "second" in units_hint:
                # treat as seconds offset
                try:
                    dt = ref + timedelta(seconds=float(s))
                    out.append(np.datetime64(np.datetime64(dt)))
                except OverflowError:
                    out.append(np.datetime64("NaT"))
                continue
            if "day" in units_hint or "days" in units_hint:
                try:
                    dt = ref + timedelta(days=float(s))
                    out.append(np.datetime64(np.datetime64(dt)))
                except OverflowError:
                    out.append(np.datetime64("NaT"))
                continue
        except Exception:
            # ignore units hint parse errors and fall back to heuristic
            pass

        # heuristic: small numbers -> days, large but < threshold -> seconds
        try:
            if abs(s) > 10000:
                # likely seconds
                dt = ref + timedelta(seconds=float(s))
            else:
                # likely days
                dt = ref + timedelta(days=float(s))
            out.append(np.datetime64(np.datetime64(dt)))
        except OverflowError:
            out.append(np.datetime64("NaT"))
        except Exception:
            out.append(np.datetime64("NaT"))

    return np.array(out, dtype="datetime64[ns]")


def decode_str_array(arr):
    """Convert bytes/np.bytes_ arrays to python strings where appropriate."""
    if arr is None:
        return None
    try:
        # numpy array of bytes
        if isinstance(arr, np.ndarray) and arr.dtype.kind == "S":
            return np.array([x.decode("utf-8", "ignore") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr])
        # object dtype might contain bytes
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            return np.array([x.decode("utf-8", "ignore") if isinstance(x, (bytes, np.bytes_)) else (str(x) if x is not None else None) for x in arr])
        # regular unicode array or scalar
        return arr
    except Exception:
        try:
            return arr.astype(str)
        except Exception:
            return arr


def safe_get_var(ds, name):
    """Return variable values or None if not present, with strings decoded."""
    if name in ds.variables:
        v = ds[name].values
        return decode_str_array(v)
    return None


def process_file(nc_path, profile_rows, level_rows, next_profile_global_index):
    try:
        ds = xr.open_dataset(nc_path, mask_and_scale=False, decode_times=False)
    except Exception as e:
        print(f"Warning: failed to open {nc_path}: {e}")
        return next_profile_global_index

    # prefer sizes to dims to avoid FutureWarning
    sizes = getattr(ds, "sizes", None) or {}
    # load attrs safely
    attrs = getattr(ds, "attrs", {})
    ref_str = attrs.get("REFERENCE_DATE_TIME") or None
    reference_dt = parse_reference_date(str(ref_str)) if ref_str else None

    # read profile-level variables (1D arrays)
    lats = safe_get_var(ds, "LATITUDE")
    lons = safe_get_var(ds, "LONGITUDE")
    platforms = safe_get_var(ds, "PLATFORM_NUMBER")
    cycles = safe_get_var(ds, "CYCLE_NUMBER")
    juld = safe_get_var(ds, "JULD")
    juld_units = ds["JULD"].attrs.get("units") if "JULD" in ds.variables and hasattr(ds["JULD"], "attrs") else None
    juld_dt = juld_to_datetime(juld if juld is not None else np.array([]), reference_dt, juld_units)

    directions = safe_get_var(ds, "DIRECTION")
    data_modes = safe_get_var(ds, "DATA_MODE")
    project_names = safe_get_var(ds, "PROJECT_NAME")

    # level vars: might be shape (N_PROF, N_LEVELS) or (N_LEVELS, N_PROF)
    PRES = safe_get_var(ds, "PRES")
    TEMP = safe_get_var(ds, "TEMP")
    PSAL = safe_get_var(ds, "PSAL")

    # determine number of profiles
    n_prof = 0
    if lats is not None:
        n_prof = int(np.shape(lats)[0])
    elif "N_PROF" in sizes:
        n_prof = int(sizes.get("N_PROF", 0))
    elif juld is not None:
        n_prof = int(np.shape(juld)[0])
    else:
        # last resort: try shape of PRES first dimension
        if PRES is not None and hasattr(PRES, "shape"):
            n_prof = int(PRES.shape[0])
        else:
            n_prof = 0

    # determine level shape helpers (we will adapt when iterating)
    def ensure_2d(var, n_prof):
        """Return var as (n_prof, n_levels) or None. If shape matches (n_prof, n_levels) keep.
           If it's transposed (n_levels, n_prof) transpose it. If var is scalar or 1D -> None.
        """
        if var is None:
            return None
        arr = np.array(var)
        if arr.ndim == 2:
            if arr.shape[0] == n_prof:
                return arr
            if arr.shape[1] == n_prof:
                return arr.T
            # shapes don't match; if one dimension equals 1, we try to broadcast
            if arr.shape[0] == 1 and n_prof > 1:
                return np.repeat(arr, n_prof, axis=0)
            # unclear mapping -> return as-is but may fail later
            return arr
        # if 1D and length == n_prof, treat as single-level
        if arr.ndim == 1 and arr.shape[0] == n_prof:
            return arr.reshape(n_prof, 1)
        return None

    PRES2 = ensure_2d(PRES, n_prof)
    TEMP2 = ensure_2d(TEMP, n_prof)
    PSAL2 = ensure_2d(PSAL, n_prof)

    # process each profile
    for p in range(n_prof):
        profile_global_id = next_profile_global_index
        next_profile_global_index += 1

        # helpers to safely index profile-level arrays
        def safe_idx(arr, idx):
            try:
                if arr is None:
                    return None
                # numpy strings might be b'' -> convert to python str
                val = arr[idx]
                if isinstance(val, (bytes, np.bytes_)):
                    return val.decode("utf-8", "ignore")
                return val
            except Exception:
                return None

        juld_val = None
        if juld_dt is not None and p < len(juld_dt):
            try:
                # pd.Timestamp can accept numpy.datetime64('NaT')
                ts = pd.Timestamp(juld_dt[p])
                juld_val = ts if not pd.isna(ts) else None
            except Exception:
                juld_val = None

        row = {
            "profile_global_id": int(profile_global_id),
            "source_file": os.path.basename(nc_path),
            "profile_index_in_file": int(p),
            "platform_number": safe_idx(platforms, p) if platforms is not None else None,
            "cycle_number": int(safe_idx(cycles, p)) if cycles is not None and safe_idx(cycles, p) is not None else None,
            "latitude": float(safe_idx(lats, p)) if lats is not None and safe_idx(lats, p) is not None and not np.isnan(float(safe_idx(lats, p))) else None,
            "longitude": float(safe_idx(lons, p)) if lons is not None and safe_idx(lons, p) is not None and not np.isnan(float(safe_idx(lons, p))) else None,
            "juld": juld_val,
            "direction": safe_idx(directions, p) if directions is not None else None,
            "data_mode": safe_idx(data_modes, p) if data_modes is not None else None,
            "project_name": safe_idx(project_names, p) if project_names is not None else None,
        }
        profile_rows.append(row)

        # build levels for this profile if we have level arrays
        if PRES2 is None or TEMP2 is None or PSAL2 is None:
            # missing one of the essential level variables -> skip level rows for safety
            continue

        # ensure we don't index out of bounds for inconsistent level counts
        n_levels = PRES2.shape[1] if PRES2 is not None else 0
        for lvl in range(n_levels):
            try:
                pres = PRES2[p, lvl]
            except Exception:
                pres = np.nan
            try:
                temp = TEMP2[p, lvl] if TEMP2 is not None and lvl < TEMP2.shape[1] else np.nan
            except Exception:
                temp = np.nan
            try:
                psal = PSAL2[p, lvl] if PSAL2 is not None and lvl < PSAL2.shape[1] else np.nan
            except Exception:
                psal = np.nan

            # skip rows that are completely missing
            if (pres is None or (isinstance(pres, float) and np.isnan(pres))) and \
               (temp is None or (isinstance(temp, float) and np.isnan(temp))) and \
               (psal is None or (isinstance(psal, float) and np.isnan(psal))):
                continue

            level_rows.append({
                "profile_global_id": int(profile_global_id),
                "level_index": int(lvl),
                "pres_dbar": float(pres) if pres is not None and not (isinstance(pres, float) and np.isnan(pres)) else None,
                "temp_degC": float(temp) if temp is not None and not (isinstance(temp, float) and np.isnan(temp)) else None,
                "psal_psu": float(psal) if psal is not None and not (isinstance(psal, float) and np.isnan(psal)) else None,
            })

    try:
        ds.close()
    except Exception:
        pass

    return next_profile_global_index


def main():
    files = sorted(glob.glob(str(RAW_DIR / "*.nc")))
    if len(files) == 0:
        print("No .nc files in", RAW_DIR)
        return

    profile_rows = []
    level_rows = []
    next_profile_index = 0

    for f in tqdm(files, desc="Processing .nc files"):
        try:
            next_profile_index = process_file(f, profile_rows, level_rows, next_profile_index)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    profiles_df = pd.DataFrame(profile_rows)
    levels_df = pd.DataFrame(level_rows)

    if not profiles_df.empty:
        # add year/month columns for partitioning (safe handling of None/NaT)
        profiles_df["juld"] = pd.to_datetime(profiles_df["juld"], errors="coerce")
        profiles_df["year"] = profiles_df["juld"].dt.year.fillna(0).astype("Int64")
        profiles_df["month"] = profiles_df["juld"].dt.month.fillna(0).astype("Int64")
        profiles_path = OUT_DIR / "profiles.parquet"
        profiles_df.to_parquet(profiles_path, index=False)
        print("Wrote", profiles_path, "rows:", len(profiles_df))
    else:
        print("No profiles extracted")

    if not levels_df.empty:
        levels_path = OUT_DIR / "levels.parquet"
        levels_df.to_parquet(levels_path, index=False)
        print("Wrote", levels_path, "rows:", len(levels_df))
    else:
        print("No level rows extracted")


if __name__ == "__main__":
    main()
