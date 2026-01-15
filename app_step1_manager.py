"""
STEP 1: DATA UPLOAD & CLASSIFICATION (Manager Agent)

Purpose:
- User uploads Excel / CSV files
- App profiles the dataset (columns, datatypes, samples)
- Gemini Manager Agent classifies dataset type
- Suggests possible analyses & key fields

This file DOES NOT run queries.
It only understands the data.
"""

# ==========================================================
# IMPORTS
# ==========================================================

import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_string_dtype,
)

# ==========================================================
# STREAMLIT SETUP
# ==========================================================

st.set_page_config(
    page_title="Data Analyser — Step 1",
    layout="wide"
)

st.title("Step 1: Upload & Classify Dataset")

# ==========================================================
# LOAD API KEY
# ==========================================================

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Add it to .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.5-flash"

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def detect_dtype(series: pd.Series) -> str:
    """
    Detects simplified datatype for LLM understanding
    """
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_numeric_dtype(series):
        return "number"
    if is_string_dtype(series):
        return "text"
    return "unknown"


def sample_rows_safe(df: pd.DataFrame, n=5, max_cols=20):
    """
    Returns small, trimmed sample rows to avoid token overflow
    """
    slim_df = df.iloc[:n, :max_cols].copy()

    def truncate(value):
        text = str(value)
        return text if len(text) <= 100 else text[:97] + "..."

    records = []
    for _, row in slim_df.iterrows():
        records.append({col: truncate(val) for col, val in row.items()})

    return records


def build_dataset_profile(df: pd.DataFrame, filename: str) -> dict:
    """
    Converts dataframe into a compact JSON profile
    suitable for Gemini classification
    """
    columns_info = []

    for col in df.columns[:50]:
        series = df[col]
        columns_info.append({
            "name": str(col),
            "dtype": detect_dtype(series),
            "null_ratio": round(float(series.isna().mean()), 4),
            "examples": series.dropna().astype(str).head(3).tolist()
        })

    return {
        "file_name": filename,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns_info,
        "sample_rows": sample_rows_safe(df)
    }

# ==========================================================
# GEMINI MANAGER AGENT PROMPT
# ==========================================================

MANAGER_PROMPT = """
You are a Manager Agent in a data analysis system.

You will receive a compact JSON profile of a dataset.

Your tasks:
1. Classify dataset_type (choose ONE):
   ["hsn_sales_register","export_register","import_register",
    "production_log","inventory_register","party_master","other"]

2. Give a short reason.

3. Suggest up to 6 possible analyses (business questions).

4. Suggest key fields (dates, ids, amounts, quantities).

Return STRICT JSON with keys:
dataset_type, confidence, reason, possible_analyses, key_fields

Return JSON only. No explanations.
"""

# ==========================================================
# GEMINI CLASSIFICATION FUNCTION
# ==========================================================

def classify_dataset(profile: dict) -> dict:
    """
    Sends dataset profile to Gemini Manager Agent
    """
    model = genai.GenerativeModel(MODEL_NAME)
    payload = json.dumps(profile, ensure_ascii=False)

    prompt = MANAGER_PROMPT + "\n\nPROFILE_JSON:\n" + payload

    try:
        response = model.generate_content(prompt)
        text = response.text or "{}"
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    except Exception as e:
        return {
            "dataset_type": "other",
            "confidence": 0,
            "reason": f"LLM error: {e}",
            "possible_analyses": [],
            "key_fields": []
        }

# ==========================================================
# STREAMLIT UI
# ==========================================================

st.markdown(
    "Upload **Excel or CSV files**. "
    "The Manager Agent will classify the dataset and suggest analyses."
)

uploaded_files = st.file_uploader(
    "Upload files",
    type=["xlsx", "xls", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        try:
            if file.name.lower().endswith(("xlsx", "xls")):
                df = pd.read_excel(file, nrows=2000)
            else:
                df = pd.read_csv(file, nrows=2000)

        except Exception as e:
            st.error(f"Failed to read {file.name}: {e}")
            continue

        with st.expander(f"Preview: {file.name}", expanded=False):
            st.write(f"Rows: {len(df)} | Columns: {len(df.columns)}")
            st.dataframe(df.head(10))

        profile = build_dataset_profile(df, file.name)

        with st.spinner("Classifying dataset with Manager Agent..."):
            result = classify_dataset(profile)

        left, right = st.columns([2, 3])

        with left:
            st.subheader("Dataset Classification")
            st.write("**Type:**", result.get("dataset_type"))
            st.write("**Confidence:**", result.get("confidence"))
            st.write("**Reason:**", result.get("reason"))

        with right:
            st.subheader("Suggested Analyses")
            for a in result.get("possible_analyses", []):
                st.write("-", a)

            st.subheader("Key Fields")
            st.write(", ".join(result.get("key_fields", [])))

    st.success("Step 1 complete ✅")

else:
    st.info("Upload at least one file to begin.")
