# The Silent Surrender: Post-Crisis Lending 2007–2017

An end-to-end data engineering and visualization project analyzing **HMDA mortgage lending data from 2007–2017** across California, Florida, Nevada, Ohio, and Texas.

The project transforms large raw HMDA datasets into clean, analysis-ready datasets using Python and presents the results through an interactive Power BI dashboard.

## Tech Stack

* **Python** — Pandas, NumPy
* **Data Engineering** — ETL, data cleaning, feature engineering, validation
* **Visualization** — Power BI
* **Data** — CFPB HMDA historical lending data

## What I Built

* Developed a reusable Python pipeline for processing large CSV and ZIP-based HMDA datasets.
* Filtered and cleaned mortgage applications across **5 states and 2007–2017**.
* Standardized loan types, purposes, and application outcomes.
* Calculated **denial, origination, and withdrawal rates**.
* Analyzed applicant income and government-backed lending trends.
* Created a custom **Surrender Index** measuring withdrawals relative to withdrawals plus denials.
* Generated **8 visualization-ready datasets** for the Power BI dashboard.

## Pipeline

```text
Raw HMDA Data
     ↓
File Discovery & Loading
     ↓
Cleaning & Filtering
     ↓
Feature Engineering
     ↓
Validation
     ↓
8 Analysis-Ready CSVs
     ↓
Power BI Dashboard
```

## Run

```bash
pip install pandas numpy
python hmda_pipeline.py --data_dir ./data --output_dir ./output
```

## Key Skills

**Python · Pandas · NumPy · ETL · Data Engineering · Data Analysis · Power BI · Data Visualization · Feature Engineering**
