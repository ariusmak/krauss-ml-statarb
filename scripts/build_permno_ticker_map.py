"""
Fetch the most recent ticker for each permno in our universe from CRSP stocknames (WRDS).
Saves a JSON dict {permno: ticker} to data/processed/permno_ticker_map.json
"""

import json
import pandas as pd
import wrds

# Connect to WRDS
db = wrds.Connection(wrds_username="ariusmak")

# Get the most recent ticker for each permno
query = """
SELECT DISTINCT ON (permno) permno, ticker, comnam
FROM crsp.stocknames
WHERE permno IS NOT NULL AND ticker IS NOT NULL
ORDER BY permno, nameenddt DESC NULLS LAST
"""
df = db.raw_sql(query)
db.close()
print(f"Fetched {len(df)} permno-ticker rows from CRSP stocknames")

# Filter to our universe
dr = pd.read_parquet("data/processed/daily_returns.parquet")
our_permnos = set(dr["permno"].unique())
mapping = df[df["permno"].isin(our_permnos)].copy()
print(f"Matched {len(mapping)} of {len(our_permnos)} permnos in our universe")

# Build and save dict
ticker_dict = {str(int(r.permno)): r.ticker for _, r in mapping.iterrows()}

out_path = "data/processed/permno_ticker_map.json"
with open(out_path, "w") as f:
    json.dump(ticker_dict, f, indent=2)
print(f"Saved to {out_path}")

# Report missing
missing = our_permnos - set(mapping["permno"])
if missing:
    print(f"\nMissing permnos ({len(missing)}): {missing}")
else:
    print("\nAll permnos matched.")
