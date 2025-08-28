"""Download ARGO NetCDF files for a month (placeholder)."""
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--month", type=int, required=True)
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()
    print("TODO: implement download. Saving to", args.out)
