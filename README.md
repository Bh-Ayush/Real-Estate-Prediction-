# Real-Estate-Prediction-
Machine Learning Model for Real Estate Price Prediction by State 

## Quick Start

```bash
# 1. Convert your dataset to Parquet (10x smaller, faster loading)
python convert_to_parquet.py your_realtor_data.csv

# 2. Run the dashboard
streamlit run real_estate_dashboard.py

# 3. Upload the generated .parquet file in the sidebar
```


### Supported File Formats
- **Parquet** (.parquet, .parq) - ⭐ Recommended for large files
- **CSV** (.csv) - Works for files <500 MB
- **ZIP** (.zip) - Archives containing CSV or Parquet files

## Usage Guide

### Running the Streamlit App

```bash
streamlit run real_estate_dashboard.py
```

Then upload your dataset using the sidebar file uploader.

### Converting Large CSV to Parquet

For large CSV files (>100 MB):

```bash
python convert_to_parquet.py realtor_data.csv realtor_data.parquet
```

**Example Output:**
```
Reading CSV: realtor_data.csv
✓ Loaded 500,000 rows × 15 columns
Converting to Parquet: realtor_data.parquet
✓ Conversion complete!
  CSV size:      250.00 MB
  Parquet size:  18.50 MB
  Compression:   92.6% smaller
```

### Interactive Dataset Setup

For a guided setup with validation:

```bash
python setup_dataset.py realtor_data.csv
```

This will:
- ✓ Validate your dataset structure
- ✓ Show file size and compression potential
- ✓ Optionally convert to Parquet
- ✓ Provide next steps

### Running Self-Test (No Streamlit UI)

Quick validation without the web interface:

```bash
python real_estate_dashboard.py --self-test
```

## Features

### Dashboard Pages
1. **Executive Summary** - Key statistics and distributions
2. **Exploratory Data Analysis** - Distributions, correlations, relationships
3. **Geographic Analysis** - State and city-level insights
4. **Price Prediction Engine** - ML model predictions with 7 algorithms
5. **Market Segmentation** - K-Means clustering analysis
6. **Model Performance** - Compare model metrics (R², RMSE, MAE, MAPE)
7. **Feature Importance** - Feature impact analysis
8. **Bias & Fairness Analysis** - Geographic and price range bias detection
9. **Market Volatility** - Risk assessment by state and city

### Supported Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

## File Size Reference

| Dataset Size | CSV Format | Parquet Format | Upload Speed |
|---|---|---|---|
| 10,000 rows | ~1 MB | ~0.1 MB | <1s |
| 100,000 rows | ~10 MB | ~1 MB | 1-2s |
| 500,000 rows | ~50 MB | ~5 MB | 2-5s |
| 1,000,000 rows | ~100 MB | ~10 MB | 5-10s |
| 5,000,000 rows | ~500 MB | ~50 MB | 30-60s |

## Handling Very Large Datasets

If your dataset is >100,000 rows:
1. Convert to Parquet for faster upload
2. Dashboard offers optional row sampling
3. Use sampling slider to select subset for analysis
4. Results remain statistically valid


## Documentation

- **DATASET_GUIDE.md** - Comprehensive guide to uploading and preparing data
- **IMPLEMENTATION_SUMMARY.md** - Technical details of improvements
- **requirements.txt** - Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Performance Tips

1. **Use Parquet** for large datasets (10x faster loading)
2. **Pre-process data** - Remove unused columns before conversion
3. **Sample large datasets** - Use the in-app sampling feature
4. **Update frequently** - Parquet compression works better on fresh data

## License & Attribution

Real Estate Price Prediction Dashboard
- ML Models: scikit-learn, XGBoost, LightGBM
- Data Visualization: Plotly
- Web Framework: Streamlit

---

**Need help?** See `DATASET_GUIDE.md` for detailed instructions on uploading your realtor dataset.
