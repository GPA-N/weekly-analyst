"""Convert the UCI Online Retail II workbook (https://archive.ics.uci.edu/dataset/502/online+retail+ii, sheet "Year 2009-2010") to the
CSV the agent reads, keeping the sheet's own words where it can. Not part of the dataset: Country is reduced to the 4 biggest plus
"Other", CustomerType is registered or guest (no Customer ID), Revenue is Quantity x Price, Returned is a cancellation invoice (Invoice
starts with C). The date column is written as an ISO date because the agent requires that. Reads an untrusted spreadsheet; needs pandas and openpyxl."""
import os, subprocess, sys, zipfile
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "/content/xl3"
os.makedirs(root, exist_ok=True)
z = f"{root}/online_retail_ii.zip"
if not os.path.exists(z):
    r = subprocess.run(["curl", "-sL", "-o", z, "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"], capture_output=True, text=True)
    print("curl rc", r.returncode, os.path.getsize(z), flush=True)
zipfile.ZipFile(z).extractall(root)
x = [p for p in os.listdir(root) if p.endswith(".xlsx")][0]
print("xlsx", x, os.path.getsize(f"{root}/{x}"), flush=True)
d = pd.read_excel(f"{root}/{x}", sheet_name=0)
print(d.shape, list(d.columns), flush=True)
d = d.dropna(subset=["InvoiceDate", "Quantity", "Price"])
top = d["Country"].value_counts().index[:4]
out = pd.DataFrame({
    "InvoiceDate": pd.to_datetime(d["InvoiceDate"]).dt.strftime("%Y-%m-%d"),
    "Country": d["Country"].where(d["Country"].isin(top), "Other").str.replace(" ", "_"),
    "CustomerType": d["Customer ID"].notna().map({True: "registered", False: "guest"}),
    "Units": d["Quantity"].abs(),
    "Revenue": (d["Quantity"].abs() * d["Price"]).round(2),
    "Returned": d["Invoice"].astype(str).str.startswith("C").astype(int),
}).sort_values("InvoiceDate")
os.makedirs(f"{root}/data", exist_ok=True)
out.to_csv(f"{root}/data/sales.csv", index=False)
w = pd.to_datetime(out["InvoiceDate"]).dt.to_period("W-SUN").value_counts().sort_index()
print(out.shape, out["InvoiceDate"].min(), out["InvoiceDate"].max(), "weeks", len(w), "rows/week median", int(w.median()), flush=True)
print(out.groupby(["Country", "CustomerType"]).size().to_dict(), flush=True)
