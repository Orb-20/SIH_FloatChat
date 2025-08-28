# scripts/parquet_to_postgres.py
import os
import sys
import time
from io import StringIO
from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# load .env
load_dotenv()

# Postgres connection env (defaults can be changed)
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB = os.getenv("POSTGRES_DB", "argo")
PG_USER = os.getenv("POSTGRES_USER", "agro")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "agro")

PROFILES_PARQUET = os.getenv("PROFILES_PARQUET", "data/interim/profiles.parquet")
LEVELS_PARQUET = os.getenv("LEVELS_PARQUET", "data/interim/levels.parquet")

# chunk size for level inserts (if extremely large)
LEVEL_CHUNK_ROWS = int(os.getenv("LEVEL_CHUNK_ROWS", "200000"))


def connect():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )


def table_exists(cur, table_name):
    cur.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s)",
        (table_name,),
    )
    return cur.fetchone()[0]


def backup_table(cur, table_name):
    """Rename existing table to a timestamped backup if it exists."""
    if not table_exists(cur, table_name):
        return None
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    new_name = f"{table_name}_backup_{ts}"
    cur.execute(sql.SQL("ALTER TABLE {} RENAME TO {}").format(
        sql.Identifier(table_name),
        sql.Identifier(new_name)
    ))
    return new_name


def create_schema(cur):
    """Create PostGIS extension and the profiles & levels tables with intended schema."""
    # ensure PostGIS exists
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")

    # profiles table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            profile_id BIGINT PRIMARY KEY,
            source_file TEXT,
            profile_index_in_file INTEGER,
            platform_number TEXT,
            cycle_number INTEGER,
            juld TIMESTAMP WITH TIME ZONE,
            latitude DOUBLE PRECISION,
            longitude DOUBLE PRECISION,
            direction TEXT,
            data_mode TEXT,
            project_name TEXT,
            geom GEOGRAPHY(POINT, 4326)
        );
        """
    )

    # levels table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS levels (
            level_id BIGSERIAL PRIMARY KEY,
            profile_id BIGINT NOT NULL,
            level_index INTEGER,
            pres_dbar DOUBLE PRECISION,
            temp_degC DOUBLE PRECISION,
            psal_psu DOUBLE PRECISION,
            FOREIGN KEY (profile_id) REFERENCES profiles(profile_id) ON DELETE CASCADE
        );
        """
    )


def df_to_copy_stringio(df, columns, header=True):
    """
    Convert df[columns] to CSV in memory (StringIO) suitable for COPY ... FROM STDIN WITH CSV.
    Uses ISO8601 for timestamps via .to_csv default formatting.
    """
    buf = StringIO()
    df.to_csv(buf, columns=columns, index=False, header=header, date_format="%Y-%m-%dT%H:%M:%S%z")
    buf.seek(0)
    return buf


def clean_profiles_df(df):
    # ensure required columns exist and have safe types
    # profile_global_id -> profile_id
    if "profile_global_id" not in df.columns:
        raise ValueError("profiles parquet missing 'profile_global_id' column")
    df = df.copy()

    # juld -> ensure datetime or NaT
    if "juld" in df.columns:
        df["juld"] = pd.to_datetime(df["juld"], errors="coerce", utc=True)
    else:
        df["juld"] = pd.NaT

    # coerce numeric lat/lon
    for c in ("latitude", "longitude"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = pd.NA

    # text fields: convert bytes / numpy objects to strings
    text_cols = ["source_file", "platform_number", "direction", "data_mode", "project_name"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else (str(x) if pd.notna(x) else None))
        else:
            df[c] = None

    # enforce profile_id integer
    df["profile_id"] = df["profile_global_id"].astype("Int64")

    # optional: trim direction / data_mode to 1 char
    df["direction"] = df["direction"].astype("string").str[:1]
    df["data_mode"] = df["data_mode"].astype("string").str[:1]

    return df


def load_profiles(conn, cur, profiles_df):
    print("Preparing profiles for load...")
    df = clean_profiles_df(profiles_df)

    # create staging temp table (keeps until session ends)
    cur.execute("DROP TABLE IF EXISTS tmp_profiles;")
    cur.execute(
        """
        CREATE TEMP TABLE tmp_profiles (
            profile_id BIGINT,
            source_file TEXT,
            profile_index_in_file INTEGER,
            platform_number TEXT,
            cycle_number INTEGER,
            juld TIMESTAMP WITH TIME ZONE,
            latitude DOUBLE PRECISION,
            longitude DOUBLE PRECISION,
            direction TEXT,
            data_mode TEXT,
            project_name TEXT
        );
        """
    )

    # select columns order
    cols = [
        "profile_id",
        "source_file",
        "profile_index_in_file",
        "platform_number",
        "cycle_number",
        "juld",
        "latitude",
        "longitude",
        "direction",
        "data_mode",
        "project_name",
    ]

    # write to in-memory buffer and COPY
    buf = df_to_copy_stringio(df, cols, header=True)
    print("COPYing profiles into temp table...")
    cur.copy_expert("COPY tmp_profiles FROM STDIN WITH CSV HEADER", buf)

    # upsert into final table: only set geom when lat/lon are not null
    print("Upserting into profiles table...")
    cur.execute(
        """
        INSERT INTO profiles (
            profile_id, source_file, profile_index_in_file,
            platform_number, cycle_number, juld, latitude, longitude,
            direction, data_mode, project_name, geom
        )
        SELECT
            profile_id, source_file, profile_index_in_file,
            platform_number, cycle_number, juld, latitude, longitude,
            direction, data_mode, project_name,
            CASE
              WHEN longitude IS NOT NULL AND latitude IS NOT NULL
              THEN ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
              ELSE NULL
            END
        FROM tmp_profiles
        ON CONFLICT (profile_id) DO UPDATE
          SET source_file = EXCLUDED.source_file,
              profile_index_in_file = EXCLUDED.profile_index_in_file,
              platform_number = EXCLUDED.platform_number,
              cycle_number = EXCLUDED.cycle_number,
              juld = EXCLUDED.juld,
              latitude = EXCLUDED.latitude,
              longitude = EXCLUDED.longitude,
              direction = EXCLUDED.direction,
              data_mode = EXCLUDED.data_mode,
              project_name = EXCLUDED.project_name,
              geom = EXCLUDED.geom;
        """
    )
    conn.commit()
    print("Profiles loaded / upserted.")


def clean_levels_df(df):
    if "profile_global_id" not in df.columns:
        raise ValueError("levels parquet missing 'profile_global_id' column")
    df = df.copy()
    df["profile_id"] = df["profile_global_id"].astype("Int64")
    # ensure floats for measurements
    for c in ("pres_dbar", "temp_degC", "psal_psu"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = pd.NA
    if "level_index" not in df.columns:
        df["level_index"] = pd.NA
    return df[["profile_id", "level_index", "pres_dbar", "temp_degC", "psal_psu"]]


def load_levels(conn, cur, levels_df):
    print("Preparing levels for load...")
    df = clean_levels_df(levels_df)

    # get unique profile ids from incoming levels; delete any existing levels for those profiles
    profile_ids = df["profile_id"].dropna().unique().tolist()
    if profile_ids:
        print(f"Deleting existing levels for {len(profile_ids)} profiles...")
        cur.execute("DELETE FROM levels WHERE profile_id = ANY(%s);", (profile_ids,))
        conn.commit()

    # load in chunks with COPY (to avoid extremely large buffers)
    total_rows = len(df)
    if total_rows == 0:
        print("No level rows to load.")
        return

    print(f"Loading {total_rows} level rows in chunks...")
    start = 0
    while start < total_rows:
        end = min(start + LEVEL_CHUNK_ROWS, total_rows)
        chunk = df.iloc[start:end]
        buf = df_to_copy_stringio(chunk, ["profile_id", "level_index", "pres_dbar", "temp_degC", "psal_psu"], header=False)
        # COPY without header into levels
        cur.copy_expert("COPY levels (profile_id, level_index, pres_dbar, temp_degC, psal_psu) FROM STDIN WITH CSV", buf)
        conn.commit()
        print(f"  loaded rows {start}..{end-1}")
        start = end

    print("Levels loaded.")


def main():
    print("Connecting to Postgres...")
    try:
        conn = connect()
    except Exception as e:
        print("Failed to connect to Postgres:", e)
        sys.exit(1)

    cur = conn.cursor()

    # Backup existing tables if present
    print("Checking existing tables...")
    prof_bak = backup_table(cur, "profiles")
    lev_bak = backup_table(cur, "levels")
    if prof_bak or lev_bak:
        print("Renamed existing tables:")
        if prof_bak:
            print(" - profiles ->", prof_bak)
        if lev_bak:
            print(" - levels ->", lev_bak)

    print("Creating schema...")
    create_schema(cur)
    conn.commit()

    # read parquet files
    if not os.path.exists(PROFILES_PARQUET):
        print("Profiles parquet not found:", PROFILES_PARQUET)
        conn.close()
        return
    if not os.path.exists(LEVELS_PARQUET):
        print("Levels parquet not found:", LEVELS_PARQUET)
        conn.close()
        return

    print("Reading parquet files...")
    profiles_df = pd.read_parquet(PROFILES_PARQUET)
    levels_df = pd.read_parquet(LEVELS_PARQUET)

    try:
        if not profiles_df.empty:
            print("Loading profiles:", len(profiles_df))
            load_profiles(conn, cur, profiles_df)
        else:
            print("No profiles to load.")
        if not levels_df.empty:
            print("Loading levels:", len(levels_df))
            load_levels(conn, cur, levels_df)
        else:
            print("No levels to load.")
    except Exception as e:
        print("Error during load:", e)
        conn.rollback()
        cur.close()
        conn.close()
        raise

    cur.close()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
