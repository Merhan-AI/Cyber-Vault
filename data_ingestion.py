"""
data_ingestion.py
-----------------
RAG-style data ingestion module for SIH26105.

Allows users to upload their own asset/vulnerability data in CSV, Excel,
JSON, or plain text format. The module intelligently maps column names,
fills missing fields with reasonable defaults, validates the data, and
returns a clean DataFrame ready for the risk engine pipeline.
"""

import re
import io
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Column alias mapping: maps common real-world column names to the 8
# required pipeline columns.
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "asset_id": [
        "id", "asset_id", "asset_identifier", "serial", "sr_no", "s.no",
        "s_no", "#", "sno", "serial_no", "serial_number", "asset_no",
    ],
    "asset_name": [
        "name", "asset_name", "asset", "device", "system", "host",
        "hostname", "server", "resource", "device_name", "system_name",
        "server_name", "resource_name", "machine", "machine_name",
        "component", "component_name", "service", "service_name",
    ],
    "asset_type": [
        "type", "asset_type", "category", "asset_category", "class",
        "device_type", "resource_type", "system_type", "classification",
        "kind", "group", "asset_class",
    ],
    "vulnerability_count": [
        "vulnerabilities", "vuln_count", "vulnerability_count", "cve_count",
        "open_vulns", "num_vulnerabilities", "critical_vulns", "total_vulns",
        "vuln", "vulns", "vulnerability_total", "cves", "num_vulns",
        "no_of_vulnerabilities", "number_of_vulnerabilities",
    ],
    "likelihood_pct": [
        "likelihood_percent", "likelihood_pct", "probability_pct",
        "risk_probability", "attack_probability", "probability_percent",
        "threat_probability_pct", "likelihood_percentage",
    ],
    "likelihood": [
        "likelihood", "probability", "attack_likelihood",
        "threat_probability", "risk_likelihood", "prob", "risk_prob",
        "breach_probability", "attack_prob",
    ],
    "potential_financial_impact_inr": [
        "financial_impact", "impact", "impact_inr", "cost",
        "potential_loss", "loss_amount", "financial_impact_inr",
        "damage_cost", "breach_cost", "impact_value", "financial_loss",
        "monetary_impact", "loss", "potential_financial_impact_inr",
        "potential_financial_impact", "financial_exposure", "exposure",
        "loss_inr", "cost_inr", "damage", "breach_impact",
    ],
    "criticality_weight": [
        "criticality", "criticality_weight", "importance", "priority",
        "business_criticality", "asset_criticality", "weight",
        "risk_weight", "business_impact", "severity", "critical",
        "business_importance", "asset_priority", "asset_importance",
    ],
}

# Substring hints: if a column name contains one of these substrings,
# map it to the corresponding target column. Ordered from most specific
# to least specific to avoid ambiguous matches.
_SUBSTRING_HINTS = [
    ("vuln",        "vulnerability_count"),
    ("cve",         "vulnerability_count"),
    ("likelihood",  "likelihood"),
    ("probab",      "likelihood"),
    ("financial",   "potential_financial_impact_inr"),
    ("impact",      "potential_financial_impact_inr"),
    ("breach_cost", "potential_financial_impact_inr"),
    ("loss",        "potential_financial_impact_inr"),
    ("critical",    "criticality_weight"),
    ("importan",    "criticality_weight"),
    ("severity",    "criticality_weight"),
    ("priority",    "criticality_weight"),
    ("weight",      "criticality_weight"),
    ("hostname",    "asset_name"),
    ("server",      "asset_name"),
    ("device_name", "asset_name"),
    ("machine",     "asset_name"),
    ("type",        "asset_type"),
    ("category",    "asset_type"),
    ("class",       "asset_type"),
    ("name",        "asset_name"),
]

REQUIRED_COLUMNS = [
    "asset_id", "asset_name", "asset_type", "vulnerability_count",
    "likelihood_pct", "likelihood", "potential_financial_impact_inr",
    "criticality_weight",
]


# ---------------------------------------------------------------------------
# 1. Parse uploaded file
# ---------------------------------------------------------------------------
def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Accept a Streamlit UploadedFile object. Detect file type and parse
    into a raw DataFrame.

    Supported: CSV, XLSX, XLS, JSON, TXT (tab/comma delimited).
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith((".xlsx", ".xls")):
        # Read all bytes into a BytesIO buffer for full file-like compatibility
        uploaded_file.seek(0)
        buf = io.BytesIO(uploaded_file.read())
        df = pd.read_excel(buf, engine="openpyxl")
    elif filename.endswith(".json"):
        df = pd.read_json(uploaded_file)
    elif filename.endswith(".txt"):
        # Try comma-separated first, then tab-separated
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=",")
            if len(df.columns) <= 1:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep="\t")
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep="\t")
    else:
        # Fallback: try CSV parsing
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            raise ValueError(
                f"Unsupported file format: '{filename}'. "
                f"Please upload a CSV, Excel (.xlsx/.xls), JSON, or TXT file. "
                f"Error: {e}"
            )

    # Drop completely empty rows/columns
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df


# ---------------------------------------------------------------------------
# 2. Normalize / map column names
# ---------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Intelligently map uploaded column names to the 8 required pipeline
    columns using exact alias matching and substring matching.

    Returns (normalized_df, list_of_mapping_messages).
    """
    messages = []
    df = df.copy()

    # Step 1: Clean column names — lowercase, strip, replace spaces/hyphens
    # with underscores
    clean_map = {}
    for col in df.columns:
        cleaned = str(col).strip().lower()
        cleaned = re.sub(r"[\s\-\.]+", "_", cleaned)
        cleaned = re.sub(r"[^\w]", "", cleaned)
        clean_map[col] = cleaned
    df = df.rename(columns=clean_map)

    # Step 2: Exact alias matching
    rename_map = {}
    matched_targets = set()

    for target_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns and target_col not in matched_targets:
                if alias != target_col:
                    rename_map[alias] = target_col
                    messages.append(f"✅ Mapped your column '{alias}' → '{target_col}'")
                else:
                    messages.append(f"✅ Found column '{target_col}' (exact match)")
                matched_targets.add(target_col)
                break

    df = df.rename(columns=rename_map)

    # Step 3: Substring / fuzzy matching for remaining unmapped columns
    unmapped_targets = set(REQUIRED_COLUMNS) - matched_targets
    if unmapped_targets:
        remaining_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
        for col in remaining_cols:
            if not unmapped_targets:
                break
            for substring, target in _SUBSTRING_HINTS:
                if target in unmapped_targets and substring in col:
                    df = df.rename(columns={col: target})
                    messages.append(
                        f"🔍 Fuzzy-mapped your column '{col}' → '{target}' "
                        f"(matched substring '{substring}')"
                    )
                    unmapped_targets.discard(target)
                    break

    return df, messages


# ---------------------------------------------------------------------------
# 3. Fill missing columns with reasonable defaults
# ---------------------------------------------------------------------------
def fill_missing_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    For each of the 8 required columns, if missing after normalization,
    fill with a reasonable default value. Also coerce data types.

    Returns (filled_df, list_of_fill_messages).
    """
    messages = []
    df = df.copy()
    n = len(df)

    # --- asset_id ---
    if "asset_id" not in df.columns or df["asset_id"].isna().all():
        df["asset_id"] = [f"A{i+1:03d}" for i in range(n)]
        messages.append("ℹ️ Column 'asset_id' was missing — generated as A001, A002, ...")

    # --- asset_name ---
    if "asset_name" not in df.columns or df["asset_name"].isna().all():
        # Try to use any string column as a fallback
        str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        fallback_cols = [c for c in str_cols if c not in REQUIRED_COLUMNS]
        if fallback_cols:
            df["asset_name"] = df[fallback_cols[0]].astype(str)
            messages.append(
                f"ℹ️ Column 'asset_name' was missing — used column '{fallback_cols[0]}' as asset names"
            )
        else:
            df["asset_name"] = [f"Asset {i+1}" for i in range(n)]
            messages.append("ℹ️ Column 'asset_name' was missing — filled with 'Asset 1', 'Asset 2', ...")

    # --- asset_type ---
    if "asset_type" not in df.columns or df["asset_type"].isna().all():
        df["asset_type"] = "Unknown"
        messages.append("ℹ️ Column 'asset_type' was missing — filled with 'Unknown'")

    # --- vulnerability_count ---
    if "vulnerability_count" not in df.columns or df["vulnerability_count"].isna().all():
        df["vulnerability_count"] = 5
        messages.append("ℹ️ Column 'vulnerability_count' was missing — filled with default value 5")
    df["vulnerability_count"] = pd.to_numeric(df["vulnerability_count"], errors="coerce").fillna(5).astype(int)
    df["vulnerability_count"] = df["vulnerability_count"].clip(lower=0)

    # --- likelihood and likelihood_pct (interdependent) ---
    has_likelihood = "likelihood" in df.columns and not df["likelihood"].isna().all()
    has_likelihood_pct = "likelihood_pct" in df.columns and not df["likelihood_pct"].isna().all()

    if has_likelihood and not has_likelihood_pct:
        df["likelihood"] = pd.to_numeric(df["likelihood"], errors="coerce").fillna(0.30)
        # If values look like percentages (>1), convert
        if df["likelihood"].median() > 1:
            df["likelihood"] = df["likelihood"] / 100.0
            messages.append("ℹ️ 'likelihood' values appeared to be percentages — converted to 0-1 range")
        df["likelihood"] = df["likelihood"].clip(0.01, 0.95)
        df["likelihood_pct"] = (df["likelihood"] * 100).round(2)
        messages.append("ℹ️ Column 'likelihood_pct' was missing — computed from 'likelihood'")
    elif has_likelihood_pct and not has_likelihood:
        df["likelihood_pct"] = pd.to_numeric(df["likelihood_pct"], errors="coerce").fillna(30.0)
        df["likelihood_pct"] = df["likelihood_pct"].clip(1, 95)
        df["likelihood"] = (df["likelihood_pct"] / 100).round(4)
        messages.append("ℹ️ Column 'likelihood' was missing — computed from 'likelihood_pct'")
    elif not has_likelihood and not has_likelihood_pct:
        # Estimate from vulnerability_count if available
        df["likelihood"] = (0.05 + (df["vulnerability_count"] / 25) * 0.5).clip(0.01, 0.95).round(4)
        df["likelihood_pct"] = (df["likelihood"] * 100).round(2)
        messages.append(
            "ℹ️ Columns 'likelihood' and 'likelihood_pct' were missing — "
            "estimated from vulnerability_count"
        )
    else:
        # Both exist — coerce types
        df["likelihood"] = pd.to_numeric(df["likelihood"], errors="coerce").fillna(0.30)
        if df["likelihood"].median() > 1:
            df["likelihood"] = df["likelihood"] / 100.0
        df["likelihood"] = df["likelihood"].clip(0.01, 0.95).round(4)
        df["likelihood_pct"] = pd.to_numeric(df["likelihood_pct"], errors="coerce").fillna(
            (df["likelihood"] * 100).round(2)
        )
        df["likelihood_pct"] = df["likelihood_pct"].clip(1, 95)

    # --- potential_financial_impact_inr ---
    if "potential_financial_impact_inr" not in df.columns or df["potential_financial_impact_inr"].isna().all():
        df["potential_financial_impact_inr"] = 50_00_000  # ₹50 Lakh
        messages.append(
            "ℹ️ Column 'potential_financial_impact_inr' was missing — "
            "filled with default ₹50,00,000 (₹50 Lakh)"
        )
    else:
        # Clean: strip ₹, Rs, INR, commas
        col = df["potential_financial_impact_inr"]
        if col.dtype == object:
            col = col.astype(str).str.replace(r"[₹,RsINR\s]", "", regex=True)
            messages.append("ℹ️ Converted financial impact from string format to numeric")
        df["potential_financial_impact_inr"] = pd.to_numeric(col, errors="coerce").fillna(50_00_000)
    df["potential_financial_impact_inr"] = df["potential_financial_impact_inr"].clip(lower=0).astype(int)

    # --- criticality_weight ---
    if "criticality_weight" not in df.columns or df["criticality_weight"].isna().all():
        df["criticality_weight"] = 0.5
        messages.append("ℹ️ Column 'criticality_weight' was missing — filled with default 0.50 (moderate)")
    else:
        df["criticality_weight"] = pd.to_numeric(df["criticality_weight"], errors="coerce").fillna(0.5)
        # If values look like percentages or 1-10 scale, normalize
        if df["criticality_weight"].max() > 1:
            if df["criticality_weight"].max() <= 10:
                df["criticality_weight"] = df["criticality_weight"] / 10.0
                messages.append("ℹ️ 'criticality_weight' appeared to be on a 1-10 scale — normalized to 0-1")
            elif df["criticality_weight"].max() <= 100:
                df["criticality_weight"] = df["criticality_weight"] / 100.0
                messages.append("ℹ️ 'criticality_weight' appeared to be a percentage — normalized to 0-1")
    df["criticality_weight"] = df["criticality_weight"].clip(0.1, 1.0).round(2)

    # Ensure only the required columns + any extras are present
    # Keep required columns first in order
    final_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    df = df[final_cols + extra_cols]

    return df, messages


# ---------------------------------------------------------------------------
# 4. Validate data
# ---------------------------------------------------------------------------
def validate_data(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate the processed DataFrame before sending to the pipeline.
    Returns (is_valid, list_of_warning_or_error_messages).

    Tries to be lenient: fix issues rather than rejecting data.
    """
    messages = []

    # Check: at least 1 row
    if df is None or len(df) == 0:
        return False, ["❌ The uploaded file contains no data rows."]

    # Check: required columns exist
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, [f"❌ Critical columns still missing after processing: {', '.join(missing)}"]

    # Check: likelihood range
    out_of_range = ((df["likelihood"] < 0) | (df["likelihood"] > 1)).sum()
    if out_of_range > 0:
        messages.append(f"⚠️ {out_of_range} row(s) had likelihood outside 0-1 range — clamped")

    # Check: criticality_weight range
    out_of_range = ((df["criticality_weight"] < 0) | (df["criticality_weight"] > 1)).sum()
    if out_of_range > 0:
        messages.append(f"⚠️ {out_of_range} row(s) had criticality_weight outside 0-1 range — clamped")

    # Check: financial impact positive
    negative_impact = (df["potential_financial_impact_inr"] < 0).sum()
    if negative_impact > 0:
        messages.append(f"⚠️ {negative_impact} row(s) had negative financial impact — set to 0")

    # Check: vulnerability_count non-negative
    negative_vulns = (df["vulnerability_count"] < 0).sum()
    if negative_vulns > 0:
        messages.append(f"⚠️ {negative_vulns} row(s) had negative vulnerability count — set to 0")

    # Check: NaN prevalence in key numeric columns
    for col in ["likelihood", "potential_financial_impact_inr", "criticality_weight"]:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            messages.append(f"⚠️ {nan_count} NaN value(s) in '{col}' — filled with defaults")

    if not messages:
        messages.append("✅ Data validation passed — all values within expected ranges")

    return True, messages


# ---------------------------------------------------------------------------
# 5. Master ingestion function
# ---------------------------------------------------------------------------
def ingest_user_data(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
    """
    Master function that chains all ingestion steps:
    1. parse_uploaded_file()
    2. normalize_columns()
    3. fill_missing_columns()
    4. validate_data()

    Returns (processed_df, list_of_info_messages).
    """
    all_messages = []

    # Step 1: Parse
    try:
        df = parse_uploaded_file(uploaded_file)
    except Exception as e:
        return pd.DataFrame(), [f"❌ Failed to parse file: {e}"]

    if len(df) == 0:
        return pd.DataFrame(), ["❌ The uploaded file contains no data rows."]

    all_messages.append(f"📄 Parsed {len(df)} rows and {len(df.columns)} columns from '{uploaded_file.name}'")

    # Step 2: Normalize columns
    df, norm_messages = normalize_columns(df)
    all_messages.extend(norm_messages)

    # Step 3: Fill missing columns
    df, fill_messages = fill_missing_columns(df)
    all_messages.extend(fill_messages)

    # Step 4: Validate
    is_valid, val_messages = validate_data(df)
    all_messages.extend(val_messages)

    if not is_valid:
        return pd.DataFrame(), all_messages

    return df, all_messages
