import sys
import os
import pandas as pd
import numpy as np

if len(sys.argv) < 4:
    print("Use as: python helper.py input.csv output.csv run_name")
    sys.exit(1)

infile = sys.argv[1]
outfile = sys.argv[2]
run_name = sys.argv[3]

df = pd.read_csv(infile)
df['real'] = pd.to_numeric(df['real'], errors='coerce')
df['prediction'] = pd.to_numeric(df['prediction'], errors='coerce')

mae = (df['prediction'] - df['real']).abs().mean()
rmse = np.sqrt(((df['prediction'] - df['real'])**2).mean())

new_row = pd.DataFrame([{
    "name": run_name,
    "mae": mae,
    "rmse": rmse
}])

if os.path.exists(outfile):
    existing = pd.read_csv(outfile)
    out_df = pd.concat([existing, new_row], ignore_index=True)
else:
    out_df = new_row

out_df.to_csv(outfile, index=False)
print(f"Appended results to {outfile}")
