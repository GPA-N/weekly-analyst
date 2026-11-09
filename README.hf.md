---
title: Weekly Analyst
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.14.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: On-device data analyst agent that writes a stakeholder PDF
---

# Weekly Analyst

An autonomous data analyst for a CSV that is replaced every week, built for the MiniCPM Spark Program (On-Device Agent Challenge).
It runs MiniCPM-V 4.7 (1B, full precision f16, no quantization) on CPU through llama.cpp.

When new data lands the agent decides, with no prompt, whether anything material changed. If not, it stays silent.
If so, it writes a stakeholder PDF. The model writes only the words: code computes every number, chart and table, a
separate implementation recomputes each number, and a person approves before anything is sent.

This Space uses synthetic sales data only and has no upload path. Reports are written only to the Space's local folder.
The first run downloads the 2 GB model and a report takes about 3 to 4 minutes on the free CPU.
