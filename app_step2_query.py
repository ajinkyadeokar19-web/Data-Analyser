"""
STEP 2: NATURAL LANGUAGE QUERY → SQL → RESULTS (Query Agent)

Purpose:
- Upload multiple Excel / CSV files
- Combine all data into one table
- Convert user question (English) → SQL using Gemini
- Execute SQL using DuckDB
- Display results and charts

This file ASSUMES data understanding is already done in Step 1.
"""

# ==========================================================
# IMPORTS
# ==========================================================

import os
import json
import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# ==========================================================
# STREAMLIT SETUP
# ==========================================================

st.set_page_config(
    page_title="Data Analyser — Step 2",
    layout="wide"
)

st.title("Step 2: Ask Questions on Your Data")

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
# GEMINI SQL PROMPT
# ==========================================================

SQL_PROMPT = """
You are an expert SQL generator.

Input:
1. A natural language business question
2. Table schema (column names and datatypes)

Rules:
- Table name is ALWAYS `data_table`
- Use ONLY provided columns
- Do NOT invent columns
- Write DuckDB-compatible SQL
- Return ONLY SQL (no explanation)
"""

# ==========================================================
# NL → SQL FUNCTION
# ==========================================================

def nl_to_sql(question: str, df: pd.DataFrame) -> str:
    """
    Converts natural language question into SQL using Gemini
    """
    schema = [
        {"column": col, "dtype": str(df[col].dtype)}
        for col in df.columns
    ]

    prompt = (
        SQL_PROMPT
        + "\n\nSCHEMA:\n"
        + json.dumps(schema, default=str)
        + "\n\nQUESTION:\n"
        + question
    )

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)

    sql = response.text or ""
    sql = (
        sql.replace("```sql", "")
           .replace("```", "")
           .replace("“", '"')
           .replace("”", '"')
           .strip()
    )

    return sql

# ==========================================================
# STREAMLIT UI — FILE UPLOAD
# ==========================================================

uploaded_files = st.file_uploader(
    "Upload Excel / CSV files",
    type=["xlsx", "xls", "csv"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload files to start querying.")
    st.stop()

# ==========================================================
# LOAD & COMBINE DATA
# ==========================================================

dataframes = []

for file in uploaded_files:
    try:
        if file.name.lower().endswith(("xlsx", "xls")):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        df["source_file"] = file.name
        dataframes.append(df)

        st.subheader(f"Preview: {file.name}")
        st.dataframe(df.head(5))

    except Exception as e:
        st.error(f"Failed to load {file.name}: {e}")

df_all = pd.concat(dataframes, ignore_index=True)

st.success(f"Loaded {len(df_all)} rows from {len(uploaded_files)} files")

# ==========================================================
# DUCKDB SETUP
# ==========================================================

con = duckdb.connect()
con.register("data_table", df_all)

# ==========================================================
# USER QUESTION
# ==========================================================

question = st.text_input(
    "Ask a question (e.g. 'HSN wise total sales')"
)

if st.button("Run Query"):

    if question.strip() == "":
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Generating SQL using Gemini..."):
        sql = nl_to_sql(question, df_all)

    st.subheader("Generated SQL")
    st.code(sql, language="sql")

    try:
        with st.spinner("Executing query..."):
            result_df = con.execute(sql).df()

        st.success("Query executed successfully")
        st.dataframe(result_df)

    except Exception as e:
        st.error(f"SQL execution failed: {e}")
        result_df = pd.DataFrame()

    # ======================================================
    # VISUALIZATION
    # ======================================================

    if not result_df.empty:
        numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
        text_cols = result_df.select_dtypes(exclude="number").columns.tolist()

        if numeric_cols and text_cols:
            st.subheader("Visualize Result")

            x_col = st.selectbox("X-axis", text_cols)
            y_col = st.selectbox("Y-axis", numeric_cols)

            chart_df = result_df[[x_col, y_col]].dropna()
            st.bar_chart(chart_df.set_index(x_col))
