"""Convert the UCI Online Retail sheet (https://archive.ics.uci.edu/dataset/352/online+retail, about 540,000 invoice rows, Dec 2010 to Dec 2011)
to the weekly CSV the agent reads (date, region, product, channel, units, revenue, discount, returned), the way the real-data example in
`examples/online-retail-*` was made. Run it on a machine you do not mind downloading a 23 MB file to; it reads an untrusted spreadsheet.

Mapping (our choices, not part of the dataset): region = United Kingdom / Germany / France / other; product = unit-price tercile;
channel = registered customer or guest (no CustomerID); returned = cancellation invoice (InvoiceNo starts with "C"); discount fixed at 0
because the sheet has no discount column. Quantity and revenue are absolute values. Then run the agent with `src/weekly_analyst/sim/metrics_sales.yaml`.
Output: /content/xl2/data/sales.csv (change the paths below for another machine). Needs pandas and openpyxl.
"""
import os, subprocess, zipfile
import pandas as pd
os.makedirs("/content/xl2", exist_ok=True)
z = "/content/xl2/online_retail.zip"
if not os.path.exists(z):
    r = subprocess.run(["curl", "-sL", "-o", z, "https://archive.ics.uci.edu/static/public/352/online+retail.zip"], capture_output=True, text=True)
    print("curl rc", r.returncode, os.path.getsize(z) if os.path.exists(z) else None)
zipfile.ZipFile(z).extractall("/content/xl2")
print(os.listdir("/content/xl2"))
f = [p for p in os.listdir("/content/xl2") if p.endswith(".xlsx")][0]
d = pd.read_excel(os.path.join("/content/xl2", f))
print(d.shape, list(d.columns)); print(d.head(2).to_string()[:600])
d = d.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])
ret = d["InvoiceNo"].astype(str).str.startswith("C")
g = d["Country"].where(d["Country"].isin(["United Kingdom", "Germany", "France"]), "other").str.lower().str.replace(" ", "_")
tier = pd.qcut(d["UnitPrice"].where(d["UnitPrice"] > 0), 3, labels=["low_price", "mid_price", "high_price"]).astype(str)
out = pd.DataFrame({"date": pd.to_datetime(d["InvoiceDate"]).dt.strftime("%Y-%m-%d"), "region": g, "product": tier,
                    "channel": d["CustomerID"].notna().map({True: "registered", False: "guest"}),
                    "units": d["Quantity"].abs(), "revenue": (d["Quantity"].abs() * d["UnitPrice"]).round(2), "discount": 0.0, "returned": ret.astype(int)})
out = out[out["product"] != "nan"].sort_values("date")
os.makedirs("/content/xl2/data", exist_ok=True)
out.to_csv("/content/xl2/data/sales.csv", index=False)
w = pd.to_datetime(out["date"]).dt.to_period("W-SUN").value_counts().sort_index()
print(out.shape, out["date"].min(), out["date"].max(), "weeks", len(w), "rows/week median", int(w.median()))
s = out.groupby(["region", "product", "channel"]).size()
print("segments", len(s), s.describe()[["min", "50%", "max"]].to_dict())
print("rows per segment per week (median):", int(s.median() / len(w)))
