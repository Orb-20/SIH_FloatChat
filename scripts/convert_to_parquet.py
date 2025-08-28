"""Convert NetCDF -> columnar parquet (placeholder)."""
import argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in-nc", required=True)
    p.add_argument("--out-parquet", default="data/interim/argo.parquet")
    args = p.parse_args()
    print("TODO: implement convert", args.in_nc, "->", args.out_parquet)
