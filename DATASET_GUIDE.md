# Getting Started: Upload Your Realtor Dataset

This guide explains how to resolve the "Error 413: Payload Too Large" and efficiently upload your realtor dataset to the Real-Estate-Prediction dashboard.

## The Problem

**Error 413** means your CSV file is too large to upload directly. By default, Streamlit has a 200 MB upload limit (we've increased it to 500 MB, but large files still cause issues).

## The Solution: Parquet Format

**Parquet files are ~10x smaller than CSV** while preserving all your data perfectly. This is the recommended approach.

### Step 1: One-Time Setup (Quick)

```bash
# If you have a large CSV file (>100 MB):
python convert_to_parquet.py realtor_data.csv
```

This creates `realtor_data.parquet` which is much smaller.

**Example output:**
```
Reading CSV: realtor_data.csv
✓ Loaded 500,000 rows × 15 columns
Converting to Parquet: realtor_data.parquet
✓ Conversion complete!
  CSV size:      250.00 MB
  Parquet size:  18.50 MB
  Compression:   92.6% smaller
```

### Step 2: Run the Dashboard

```bash
streamlit run real_estate_dashboard.py
```

### Step 3: Upload Your File

In the Streamlit app sidebar, click "Upload dataset" and select:
- ✅ **Your `.parquet` file** (fastest, recommended)
- ✅ CSV file (if small enough)
- ✅ ZIP archive containing either format

## Alternative: Quick Data Validation

Before uploading, you can validate your dataset:

```bash
python setup_dataset.py realtor_data.csv
```

This will:
- ✓ Check file size
- ✓ Verify column structure
- ✓ Optionally convert to Parquet interactively
- ✓ Show compression statistics

## Supported Formats

| Format | Max Size | Speed | Recommended | Notes |
|--------|----------|-------|-------------|-------|
| **Parquet** | 500 MB | ⚡⚡⚡ Fast | ✅ Yes | 10x smaller, fastest loading |
| CSV | 500 MB | ⚡ Medium | ⚠️ If small | Use Parquet for large files |
| ZIP | 500 MB | ⚡⚡ Fast | ✅ Yes | Can contain CSV or Parquet |

## What If I Still Get Error 413?

If you're still hitting size limits after converting to Parquet:

### Option A: Use Data Sampling (In-App)
When you upload a large dataset (>100k rows), the dashboard will offer optional sampling:
- Slider to select how many rows to analyze
- Still gives you accurate insights
- Much faster processing

### Option B: Pre-Process Your Data
Filter your CSV before converting:

```python
import pandas as pd

df = pd.read_csv('huge_realtor_data.csv')
# Keep only recent data
df = df[df['year'] >= 2020]
# Save and convert
df.to_csv('filtered_realtor_data.csv', index=False)
```

Then convert to Parquet normally.

## File Size Reference

Here's what typical dataset sizes look like:

```
10,000 rows   → ~1 MB CSV     → ~0.1 MB Parquet
100,000 rows  → ~10 MB CSV    → ~1 MB Parquet
500,000 rows  → ~50 MB CSV    → ~5 MB Parquet
1,000,000 rows → ~100 MB CSV  → ~10 MB Parquet
5,000,000 rows → ~500 MB CSV  → ~50 MB Parquet
```

## Troubleshooting

### Q: I get "Error 413" even with Parquet
**A:** Your Parquet file is >500 MB. Options:
1. Use sampling feature (select subset of rows)
2. Pre-filter your data to fewer records
3. Convert to Parquet with lower compression (trade speed for smaller file)

### Q: Data looks different after Parquet conversion
**A:** It's not! Parquet preserves all data types and values perfectly. It's just a different storage format.

### Q: How long does conversion take?
**A:** ~1-2 seconds per 100 MB of CSV. A 500 MB file typically takes 5-10 seconds.

### Q: Can I upload a ZIP with Parquet inside?
**A:** Yes! The dashboard accepts ZIP files containing either CSV or Parquet files. Just make sure the file extension is correct.

## Quick Reference

```bash
# Convert CSV to Parquet
python convert_to_parquet.py your_data.csv

# Or interactively validate and convert
python setup_dataset.py your_data.csv

# Then run the dashboard
streamlit run real_estate_dashboard.py
```

That's it! Your data is ready to explore. 🚀
