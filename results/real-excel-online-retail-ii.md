# Second real Excel test: UCI Online Retail II (2026-11-13)

A real `.xlsx` file the agent had not seen: the UCI "Online Retail II" workbook, sheet "Year 2009-2010" (525,461 invoice lines, 2009-12-01 to 2010-12-09, 53 weeks; 45 MB zip). Run on a Colab T4 with the `lora3` GGUF and the model's plan and draft calls sent without a schema or grammar. Weeks 2010-W25, W38, W47 and W48.

## Steps

1. **Raw workbook straight into the app** (`weekly-analyst run online_retail_II.xlsx`). The app halted at extract, and no report was written.
   Reason shown: "file is not valid UTF-8". Correct halt, wrong explanation. Fixed: it now says "this is an Excel workbook, not a CSV; export the sheet as CSV (UTF-8) first" (`not_csv`).
2. **Fresh install.** A first run with a work folder that does not exist yet crashed with `sqlite3.OperationalError: unable to open database file`. Fixed in `StateStore`, which now creates the folder.
3. **Convert the sheet to CSV** with `scripts/online_retail_ii_to_csv.py`. Column names are the sheet's own (`InvoiceDate`, `Country`, `CustomerType`, `Units`, `Revenue`, `Returned`). Our choices, not the dataset's: Country reduced to the four biggest plus `Other`, CustomerType registered or guest, Revenue = quantity x price, Returned = cancellation invoice. Metrics file: `examples/metrics_online_retail_ii.yaml`, with `date_column: InvoiceDate`.
4. **First pipeline run: 0 of 4 weeks.** Every week failed with `ColumnNotFoundError: unable to find column "date"`. The independent number check (`recompute`) and the model-path calculation (`compute`) had the column name `date` hard-coded and ignored `date_column` from the metrics file. Every earlier test used a column called `date`, so none showed it. Fixed: both take the date column from the metrics file (default `date`), with a regression test.
5. **Second run: 4 of 4 weeks produced a full report.** The model's plan and draft were valid each time without any constraint.

## Checks on one report (2010-W47)

Recomputed from the CSV with a separate polars script: total revenue 314,821.68, change against the prior week -44,366.91, return rate 0.0126, EIRE registered return rate 0.0659 (shown as 0.07) and EIRE registered revenue 2,321.56. All match the PDF.

Files: `examples/online-retail-ii-2010-w47.pdf` (and `-p1..p6.png`), `examples/online-retail-ii-model-replies.json`.

## What this does not show

- One dataset, one run per week, and one reader checked one report. It is not a quality score.
- The draft text is formulaic ("Return rate moved up in EIRE registered", the same shape for every segment). The recommended action is the same sentence each week. The numbers come from checked code, not the model.
- The agent still reads CSV only. An Excel file needs a conversion step like the script above.
- Not run on the Space.
