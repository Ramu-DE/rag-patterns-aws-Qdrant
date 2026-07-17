# data/

This folder holds the PDF used across all notebooks in this series.
The PDF itself is excluded from the repo (`.gitignore`) — download or place it here before running any notebook.

## Required file

| File | Used in | Description |
|------|---------|-------------|
| `climate.pdf` | All notebooks (01–30) | Climate science reference document used as the RAG knowledge base |

## How to get climate.pdf

Any publicly available climate / meteorology PDF works.
A good free source:

- **IPCC AR6 Summary for Policymakers** (public domain):
  https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_SPM.pdf
  Download and save as `data/climate.pdf`.

- Or use any climate / weather science PDF you already have.

## Notebook PDF path

Every notebook hardcodes the absolute path in its **Step 2 — Configuration** cell:

```python
PDF_PATH = r"C:\Users\Administrator\RAG\data\climate.pdf"
```

Update this path to match your local setup before running.
You can also set it via the `.env` file:

```
PDF_PATH=C:\Users\YourName\RAG\data\climate.pdf
```

## Other data files

| File | Notes |
|------|-------|
| `nike_2023_annual_report.txt` | Used in multi-document RAG experiments |
| `sample-movies-small.json` | Sample dataset for metadata filtering tests |
