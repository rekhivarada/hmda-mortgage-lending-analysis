"""
=============================================================================
HMDA DATA CLEANING PIPELINE — Data Grand Prix 2026
"The Silent Surrender: Post-Crisis Lending 2007-2017"

Built specifically for CFPB historic HMDA data files with structure:
  - Format   : CSV, comma-delimited, quoted fields
  - Filename : hmda_{year}_{state}_all-records_labels.csv
  - Location : Inside per-state zip files (e.g. Florida.zip)
  - Rows/file : ~1M per state per year

Target states : California, Florida, Nevada, Ohio, Texas
Output        : 8 clean CSVs ready for visualization

HOW TO RUN:
    pip install pandas numpy
    python hmda_pipeline.py --data_dir ./your_data_folder --output_dir ./output

Authors: Data Grand Prix 2026 Team
=============================================================================
"""

import os
import sys
import zipfile
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# Exactly matched to real CFPB column names confirmed from your files
# =============================================================================

TARGET_STATES = {
    "CA": "California",
    "FL": "Florida",
    "NV": "Nevada",
    "OH": "Ohio",
    "TX": "Texas",
}

YEARS = list(range(2007, 2018))

# EXACT column names from your files — confirmed from Florida 2017 sample
KEEP_COLUMNS = [
    "as_of_year",
    "state_abbr",
    "state_name",
    "action_taken",
    "action_taken_name",
    "loan_type",
    "loan_type_name",
    "loan_purpose",
    "loan_purpose_name",
    "applicant_income_000s",
    "loan_amount_000s",
    "lien_status",
    "lien_status_name",
    "msamd_name",
    "county_name",
]

# Confirmed from your actual action_taken_name values
ACTION_LABEL_MAP = {
    "Loan originated":                                      "originated",
    "Application approved but not accepted":                "approved_not_accepted",
    "Application denied by financial institution":          "denied",
    "Application withdrawn by applicant":                   "withdrawn",
    "File closed for incompleteness":                       "withdrawn",
    "Loan purchased by the institution":                    "purchased",
    "Preapproval request denied by financial institution":  "denied",
    "Preapproval request approved but not accepted":        "approved_not_accepted",
}

# Confirmed from your actual loan_type_name values
LOAN_TYPE_MAP = {
    "Conventional":       "conventional",
    "FHA-insured":        "fha",
    "VA-guaranteed":      "va",
    "FSA/RHS-guaranteed": "govt_other",
    "FSA/RHS & RHS":      "govt_other",
    "FSA/RHS":            "govt_other",
}

LOAN_PURPOSE_MAP = {
    "Home purchase":   "home_purchase",
    "Home improvement":"home_improvement",
    "Refinancing":     "refinancing",
}

# California MSA split for V14 — Inland Empire vs Bay Area/LA coastal
CA_INLAND_KEYWORDS = [
    "Riverside", "San Bernardino", "Bakersfield",
    "Stockton", "Modesto", "Fresno", "Visalia",
    "Merced", "El Centro", "Hanford", "Madera",
]

CA_COASTAL_KEYWORDS = [
    "San Francisco", "Oakland", "San Jose",
    "Los Angeles", "Anaheim", "San Diego",
    "Santa Ana", "Oxnard", "Santa Rosa",
    "Napa", "Santa Cruz", "Santa Barbara",
    "Long Beach", "Ventura",
]


# =============================================================================
# STEP 1 — FILE DISCOVERY
# =============================================================================

def discover_files(data_dir: Path) -> list:
    print(f"\n{'='*60}")
    print(f"STEP 1 — Discovering files in: {data_dir}")
    print(f"{'='*60}")

    found = []

    for fpath in sorted(data_dir.rglob("*")):
        name = fpath.name.lower()

        if name.endswith(".csv") and "hmda" in name:
            found.append(("csv", fpath))
            print(f"  [CSV] {fpath.name}")

        elif name.endswith(".zip"):
            try:
                with zipfile.ZipFile(fpath, "r") as z:
                    inner_csvs = [f for f in z.namelist() if f.endswith(".csv")]
                    if inner_csvs:
                        found.append(("zip", fpath))
                        print(f"  [ZIP] {fpath.name} — {len(inner_csvs)} CSV(s) inside")
            except Exception as e:
                print(f"  [ERR] Cannot read {fpath.name}: {e}")

    print(f"\n  Total files found: {len(found)}")
    if not found:
        print(f"\n  [ERROR] No files found in {data_dir}")
        print(f"  Put your state zip files (Florida.zip etc) in that folder")
        sys.exit(1)

    return found


# =============================================================================
# STEP 2 — LOADING
# =============================================================================

def read_hmda_csv(f) -> pd.DataFrame:
    """Read a single HMDA CSV, keeping only our columns."""
    try:
        df = pd.read_csv(
            f,
            usecols=lambda c: c in KEEP_COLUMNS,
            dtype=str,
            encoding="latin-1",
            low_memory=False,
            on_bad_lines="skip",
        )
        return df
    except Exception:
        # Fallback: read all columns then filter
        try:
            if hasattr(f, "seek"):
                f.seek(0)
            df = pd.read_csv(
                f, dtype=str, encoding="latin-1",
                low_memory=False, on_bad_lines="skip"
            )
            df.columns = [c.strip() for c in df.columns]
            keep = [c for c in KEEP_COLUMNS if c in df.columns]
            return df[keep]
        except Exception as e2:
            print(f"    [ERR] Read failed: {e2}")
            return pd.DataFrame()


def load_file(fpath: Path, file_type: str) -> pd.DataFrame:
    try:
        if file_type == "csv":
            with open(fpath, "r", encoding="latin-1") as f:
                df = read_hmda_csv(f)
            print(f"  [OK]  {fpath.name}: {len(df):,} rows")
            return df

        elif file_type == "zip":
            chunks = []
            with zipfile.ZipFile(fpath, "r") as z:
                csv_files = sorted([
                    f for f in z.namelist()
                    if f.endswith(".csv") and "hmda" in f.lower()
                ])
                if not csv_files:
                    csv_files = [f for f in z.namelist() if f.endswith(".csv")]

                for csv_name in csv_files:
                    with z.open(csv_name) as f:
                        df = read_hmda_csv(f)
                        if not df.empty:
                            chunks.append(df)
                            print(f"       {csv_name}: {len(df):,} rows")

            if not chunks:
                return pd.DataFrame()

            combined = pd.concat(chunks, ignore_index=True)
            print(f"  [OK]  {fpath.name}: {len(combined):,} total rows")
            return combined

    except Exception as e:
        print(f"  [ERR] {fpath.name}: {e}")
        return pd.DataFrame()


def load_all_files(file_list: list) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"STEP 2 — Loading {len(file_list)} file(s)")
    print(f"         This takes a few minutes — files are large")
    print(f"{'='*60}")

    chunks = []
    for i, (file_type, fpath) in enumerate(file_list, 1):
        print(f"\n  [{i}/{len(file_list)}] {fpath.name}")
        df = load_file(fpath, file_type)
        if not df.empty:
            chunks.append(df)

    if not chunks:
        print("\n[ERROR] No data loaded. Check your file locations.")
        sys.exit(1)

    master = pd.concat(chunks, ignore_index=True)
    print(f"\n  MASTER: {len(master):,} total rows loaded")
    return master


# =============================================================================
# STEP 3 — FILTER
# =============================================================================

def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"STEP 3 — Filtering to 5 states and years 2007-2017")
    print(f"{'='*60}")

    before = len(df)

    # Filter states using state_abbr (confirmed in your data as "FL", "CA" etc)
    if "state_abbr" in df.columns:
        df = df[df["state_abbr"].isin(TARGET_STATES.keys())].copy()
    elif "state_name" in df.columns:
        df = df[df["state_name"].isin(TARGET_STATES.values())].copy()
    else:
        print("  [WARN] No state column found — cannot filter states")

    # Filter years
    if "as_of_year" in df.columns:
        df["as_of_year"] = pd.to_numeric(df["as_of_year"], errors="coerce")
        df = df[df["as_of_year"].isin(YEARS)].copy()

    after = len(df)
    pct = after / before * 100 if before > 0 else 0
    print(f"  Before : {before:,}")
    print(f"  After  : {after:,} ({pct:.1f}% retained)")

    if "state_abbr" in df.columns:
        print(f"  States : {sorted(df['state_abbr'].dropna().unique().tolist())}")
    if "as_of_year" in df.columns:
        yrs = sorted(df["as_of_year"].dropna().unique().astype(int).tolist())
        print(f"  Years  : {yrs}")

    return df


# =============================================================================
# STEP 4 — CLEAN AND RECODE
# =============================================================================

def clean_and_recode(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"STEP 4 — Cleaning and recoding")
    print(f"{'='*60}")

    # Strip whitespace from all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Clean action taken — using confirmed label names from your data
    if "action_taken_name" in df.columns:
        df["action_clean"] = df["action_taken_name"].map(ACTION_LABEL_MAP)
        n_unmapped = df["action_clean"].isna().sum()
        if n_unmapped > 0:
            uniques = df.loc[df["action_clean"].isna(), "action_taken_name"].unique()
            print(f"  [WARN] {n_unmapped} unmapped action_taken_name: {uniques}")

    # Clean loan type
    if "loan_type_name" in df.columns:
        df["loan_type_clean"] = df["loan_type_name"].map(LOAN_TYPE_MAP)
        df["is_govt_backed"] = df["loan_type_clean"].isin(["fha", "va", "govt_other"])

    # Clean loan purpose
    if "loan_purpose_name" in df.columns:
        df["loan_purpose_clean"] = df["loan_purpose_name"].map(LOAN_PURPOSE_MAP)

    # Income — fix type, remove junk values
    if "applicant_income_000s" in df.columns:
        df["applicant_income_000s"] = pd.to_numeric(
            df["applicant_income_000s"], errors="coerce"
        )
        df.loc[df["applicant_income_000s"] > 9999, "applicant_income_000s"] = np.nan
        df.loc[df["applicant_income_000s"] <= 0,   "applicant_income_000s"] = np.nan

    # Loan amount
    if "loan_amount_000s" in df.columns:
        df["loan_amount_000s"] = pd.to_numeric(
            df["loan_amount_000s"], errors="coerce"
        )

    # California region flag for V14 (Inland vs Coastal contrast)
    df["ca_region"] = "other"
    if "msamd_name" in df.columns:
        is_ca = df["state_abbr"] == "CA"
        for kw in CA_INLAND_KEYWORDS:
            mask = is_ca & df["msamd_name"].str.contains(kw, na=False, case=False)
            df.loc[mask, "ca_region"] = "inland"
        for kw in CA_COASTAL_KEYWORDS:
            mask = is_ca & df["msamd_name"].str.contains(kw, na=False, case=False)
            df.loc[mask, "ca_region"] = "coastal"

    # Print recode summary
    if "action_clean" in df.columns:
        print(f"  action_clean      : {df['action_clean'].value_counts().to_dict()}")
    if "loan_type_clean" in df.columns:
        print(f"  loan_type_clean   : {df['loan_type_clean'].value_counts().to_dict()}")
    if "ca_region" in df.columns:
        ca_rc = df[df["state_abbr"] == "CA"]["ca_region"].value_counts()
        print(f"  CA regions        : {ca_rc.to_dict()}")

    return df


# =============================================================================
# STEP 5 — BUILD CHAPTER DATASETS
# =============================================================================

def build_chapter1(df: pd.DataFrame):
    """
    CHAPTER 1 — The FHA Lifeline
    Feeds: V3 (stacked area), V4 (grouped bar), V5 (multi-line), V6 (area gap)
    """
    print(f"\n  [CH1] FHA Lifeline...")

    orig = df[df["action_clean"] == "originated"].copy()
    if orig.empty:
        print("  [WARN] No originated loans found")
        return pd.DataFrame(), pd.DataFrame()

    # Detailed: count by state, year, loan type
    detail = (
        orig.groupby(["state_abbr", "state_name", "as_of_year", "loan_type_clean"])
        .size().reset_index(name="loan_count")
    )
    totals = detail.groupby(["state_abbr", "as_of_year"])["loan_count"].sum().rename("total")
    detail = detail.join(totals, on=["state_abbr", "as_of_year"])
    detail["share_pct"] = (detail["loan_count"] / detail["total"] * 100).round(2)
    detail = detail.drop(columns=["total"])

    # Summary: conventional vs govt backed
    summary = (
        orig.groupby(["state_abbr", "state_name", "as_of_year", "is_govt_backed"])
        .size().reset_index(name="loan_count")
    )
    totals2 = summary.groupby(["state_abbr", "as_of_year"])["loan_count"].sum().rename("total")
    summary = summary.join(totals2, on=["state_abbr", "as_of_year"])
    summary["share_pct"] = (summary["loan_count"] / summary["total"] * 100).round(2)
    summary["type"] = summary["is_govt_backed"].map({True: "govt_backed", False: "conventional"})
    summary = summary.drop(columns=["total", "is_govt_backed"])

    print(f"         Detail: {len(detail):,} rows | Summary: {len(summary):,} rows")
    return detail, summary


def build_chapter2(df: pd.DataFrame):
    """
    CHAPTER 2 — The Denial Gap
    Feeds: V7 (multi-line), V8 (stacked bar), V9 (interactive map), V10 (scatter)
    """
    print(f"  [CH2] Denial Gap...")

    relevant = ["originated", "denied", "approved_not_accepted", "withdrawn"]
    ch2 = df[df["action_clean"].isin(relevant)].copy()

    # Detailed counts
    detail = (
        ch2.groupby(["state_abbr", "state_name", "as_of_year", "action_clean"])
        .size().reset_index(name="count")
    )

    # Denial rate table
    pivot = detail.pivot_table(
        index=["state_abbr", "state_name", "as_of_year"],
        columns="action_clean", values="count", fill_value=0
    ).reset_index()
    pivot.columns.name = None

    for col in ["originated", "denied", "approved_not_accepted", "withdrawn"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["total_decisions"] = (
        pivot["originated"] + pivot["denied"] + pivot["approved_not_accepted"]
    )
    pivot["total_applications"] = pivot["total_decisions"] + pivot["withdrawn"]
    pivot["denial_rate_pct"] = (
        pivot["denied"] / pivot["total_decisions"].replace(0, np.nan) * 100
    ).round(2)
    pivot["origination_rate_pct"] = (
        pivot["originated"] / pivot["total_applications"].replace(0, np.nan) * 100
    ).round(2)
    pivot["withdrawal_rate_pct"] = (
        pivot["withdrawn"] / pivot["total_applications"].replace(0, np.nan) * 100
    ).round(2)

    # Income analysis for scatter V10
    income = pd.DataFrame()
    if "applicant_income_000s" in df.columns:
        income = (
            ch2.groupby(["state_abbr", "state_name", "as_of_year", "action_clean"])
            .agg(
                median_income=("applicant_income_000s", "median"),
                mean_income=("applicant_income_000s", "mean"),
                count=("applicant_income_000s", "count"),
            ).round(2).reset_index()
        )

    print(f"         Detail: {len(detail):,} rows | Rates: {len(pivot):,} rows | Income: {len(income):,} rows")
    return detail, pivot, income


def build_chapter3(df: pd.DataFrame):
    """
    CHAPTER 3 — The Silent Surrender
    Feeds: V11 (dual line), V12 (bar), V13 (small multiples), V14 (CA contrast)
    """
    print(f"  [CH3] Silent Surrender...")

    ch3 = df[df["action_clean"].isin(["denied", "withdrawn", "originated"])].copy()

    # State surrender index
    counts = (
        ch3.groupby(["state_abbr", "state_name", "as_of_year", "action_clean"])
        .size().reset_index(name="count")
    )
    pivot = counts.pivot_table(
        index=["state_abbr", "state_name", "as_of_year"],
        columns="action_clean", values="count", fill_value=0
    ).reset_index()
    pivot.columns.name = None

    for col in ["originated", "denied", "withdrawn"]:
        if col not in pivot.columns:
            pivot[col] = 0

    # THE SURRENDER INDEX — our original metric
    # = withdrawals / (withdrawals + denials)
    # 1.0 = everyone gave up themselves
    # 0.0 = everyone was officially rejected by the bank
    pivot["surrender_index"] = (
        pivot["withdrawn"] /
        (pivot["withdrawn"] + pivot["denied"]).replace(0, np.nan)
    ).round(4)

    # Crisis type classification for V13
    pivot["crisis_type"] = pd.cut(
        pivot["surrender_index"],
        bins=[0, 0.35, 0.50, 0.65, 1.01],
        labels=["bank_refusal", "mixed_bank", "mixed_surrender", "self_surrender"]
    )

    # Year over year change
    pivot = pivot.sort_values(["state_abbr", "as_of_year"])
    pivot["surrender_yoy"] = (
        pivot.groupby("state_abbr")["surrender_index"].diff().round(4)
    )

    # National trend for V11
    national = (
        ch3.groupby(["as_of_year", "action_clean"])
        .size().reset_index(name="count")
    )
    national_pivot = national.pivot(
        index="as_of_year", columns="action_clean", values="count"
    ).reset_index()
    national_pivot.columns.name = None

    # California internal contrast for V14
    ca_contrast = pd.DataFrame()
    if "ca_region" in df.columns:
        ca = df[
            (df["state_abbr"] == "CA") &
            (df["ca_region"].isin(["inland", "coastal"])) &
            (df["action_clean"].isin(["denied", "withdrawn", "originated"]))
        ].copy()

        if not ca.empty:
            ca_c = (
                ca.groupby(["ca_region", "as_of_year", "action_clean"])
                .size().reset_index(name="count")
            )
            ca_p = ca_c.pivot_table(
                index=["ca_region", "as_of_year"],
                columns="action_clean", values="count", fill_value=0
            ).reset_index()
            ca_p.columns.name = None

            for col in ["originated", "denied", "withdrawn"]:
                if col not in ca_p.columns:
                    ca_p[col] = 0

            ca_p["denial_rate_pct"] = (
                ca_p["denied"] /
                (ca_p["denied"] + ca_p["originated"]).replace(0, np.nan) * 100
            ).round(2)
            ca_p["surrender_index"] = (
                ca_p["withdrawn"] /
                (ca_p["withdrawn"] + ca_p["denied"]).replace(0, np.nan)
            ).round(4)
            ca_contrast = ca_p

    print(f"         States: {len(pivot):,} rows | National: {len(national_pivot):,} rows | CA: {len(ca_contrast):,} rows")
    return pivot, national_pivot, ca_contrast


# =============================================================================
# STEP 6 — VALIDATE
# =============================================================================

def validate(ch1_d, ch1_s, ch2_d, ch2_r, ch2_i, ch3_s, ch3_n, ch3_ca):
    print(f"\n{'='*60}")
    print(f"STEP 6 — Validation")
    print(f"{'='*60}")

    issues = []

    for df, name in [(ch1_d, "Ch1"), (ch2_r, "Ch2"), (ch3_s, "Ch3")]:
        if df is None or df.empty:
            issues.append(f"{name} is empty")
            continue
        if "state_abbr" in df.columns:
            missing = set(TARGET_STATES.keys()) - set(df["state_abbr"].unique())
            if missing:
                issues.append(f"{name} missing states: {missing}")

    if not ch2_r.empty and "as_of_year" in ch2_r.columns:
        missing_yrs = set(YEARS) - set(ch2_r["as_of_year"].unique().astype(int))
        if missing_yrs:
            issues.append(f"Ch2 missing years: {sorted(missing_yrs)}")

    if not ch2_r.empty and "denial_rate_pct" in ch2_r.columns:
        mx = ch2_r["denial_rate_pct"].max()
        mn = ch2_r["denial_rate_pct"].min()
        if mx > 85:
            issues.append(f"Denial rate suspiciously high: {mx:.1f}%")
        if mn < 3:
            issues.append(f"Denial rate suspiciously low: {mn:.1f}%")

    if issues:
        print(f"  [WARNINGS]")
        for i in issues:
            print(f"    - {i}")
    else:
        print(f"  All checks passed")

    # Key findings
    print(f"\n  KEY FINDINGS:")
    if not ch2_r.empty and "denial_rate_pct" in ch2_r.columns:
        peak = ch2_r.loc[ch2_r["denial_rate_pct"].idxmax()]
        print(f"  Peak denial    : {peak['denial_rate_pct']:.1f}% — {peak['state_abbr']} in {int(peak['as_of_year'])}")
        tx = ch2_r[ch2_r["state_abbr"] == "TX"]
        if not tx.empty:
            print(f"  TX peak denial : {tx['denial_rate_pct'].max():.1f}%")
    if not ch3_s.empty and "surrender_index" in ch3_s.columns:
        ps = ch3_s.loc[ch3_s["surrender_index"].idxmax()]
        print(f"  Peak surrender : {ps['surrender_index']:.3f} — {ps['state_abbr']} in {int(ps['as_of_year'])}")
    if not ch1_s.empty and "share_pct" in ch1_s.columns:
        g = ch1_s[ch1_s["type"] == "govt_backed"].nlargest(1, "share_pct")
        if not g.empty:
            r = g.iloc[0]
            print(f"  Peak FHA share : {r['share_pct']:.1f}% — {r['state_abbr']} in {int(r['as_of_year'])}")


# =============================================================================
# STEP 7 — EXPORT
# =============================================================================

def export(out: Path, ch1_d, ch1_s, ch2_d, ch2_r, ch2_i, ch3_s, ch3_n, ch3_ca):
    print(f"\n{'='*60}")
    print(f"STEP 7 — Exporting to {out}")
    print(f"{'='*60}")

    out.mkdir(parents=True, exist_ok=True)

    outputs = {
        "chapter1_fha_detail.csv":          ch1_d,   # feeds V3, V5
        "chapter1_fha_summary.csv":         ch1_s,   # feeds V4, V6
        "chapter2_denial_detail.csv":       ch2_d,   # feeds V8
        "chapter2_denial_rates.csv":        ch2_r,   # feeds V7, V9
        "chapter2_income_analysis.csv":     ch2_i,   # feeds V10
        "chapter3_surrender_states.csv":    ch3_s,   # feeds V12, V13
        "chapter3_surrender_national.csv":  ch3_n,   # feeds V11
        "chapter3_california_contrast.csv": ch3_ca,  # feeds V14
    }

    for fname, df in outputs.items():
        fpath = out / fname
        if df is None or (hasattr(df, "empty") and df.empty):
            print(f"  [SKIP] {fname} — empty")
        else:
            df.to_csv(fpath, index=False)
            kb = fpath.stat().st_size // 1024
            print(f"  [OK]   {fname} — {len(df):,} rows ({kb} KB)")

    print(f"\n  Upload these 8 CSV files to Claude to build the dashboard")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="HMDA Pipeline — Data Grand Prix 2026")
    parser.add_argument("--data_dir",   default="./data",   help="Folder with your HMDA zip/csv files")
    parser.add_argument("--output_dir", default="./output", help="Folder to write clean CSVs")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  HMDA PIPELINE — Data Grand Prix 2026")
    print("  'The Silent Surrender: 2007-2017'")
    print("="*60)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Run the 7-step pipeline
    files  = discover_files(data_dir)
    master = load_all_files(files)
    master = filter_data(master)
    master = clean_and_recode(master)

    print(f"\n{'='*60}")
    print(f"STEP 5 — Building chapter datasets")
    print(f"{'='*60}")

    ch1_d, ch1_s          = build_chapter1(master)
    ch2_d, ch2_r, ch2_i   = build_chapter2(master)
    ch3_s, ch3_n, ch3_ca  = build_chapter3(master)

    validate(ch1_d, ch1_s, ch2_d, ch2_r, ch2_i, ch3_s, ch3_n, ch3_ca)
    export(output_dir, ch1_d, ch1_s, ch2_d, ch2_r, ch2_i, ch3_s, ch3_n, ch3_ca)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  8 clean CSVs are ready in: {output_dir.resolve()}")
    print(f"  Upload them to Claude — we build the dashboard next")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
