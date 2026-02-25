import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

# -----------------------------------------------------------------------------
# App Config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Database Chatbot", layout="wide")

session = get_active_session()

st.title("🧠 Snowflake Cortex AI - Healthcare Chatbot")

# -----------------------------------------------------------------------------
# Get Database Schema Dynamically
# -----------------------------------------------------------------------------
@st.cache_data
def get_schema():

    schema_query = """
    SELECT
        table_schema,
        table_name,
        column_name
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE table_schema NOT IN ('INFORMATION_SCHEMA')
    ORDER BY table_schema, table_name
    """

    df = session.sql(schema_query).to_pandas()

    schema_text = ""

    for table in df["TABLE_NAME"].unique():
        cols = df[df["TABLE_NAME"] == table]["COLUMN_NAME"].tolist()
        schema_text += f"\nTable: {table}\nColumns: {', '.join(cols)}\n"

    return schema_text


schema_info = get_schema()

# -----------------------------------------------------------------------------
# Generate SQL using Cortex
# -----------------------------------------------------------------------------
def generate_sql(question):

    prompt = f"""
You are an expert Snowflake SQL generator.

Database Schema:
{schema_info}

User Question:
{question}

Instructions:

1. Find relevant tables automatically.
2. Identify patient name columns even if named:
   - full_name
   - first_name
   - last_name
   - patient_name
3. Join tables if needed.
4. Return only valid Snowflake SQL.
5. Limit result to 20 rows.

Return SQL only.
"""

    sql = session.sql(f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            'llama3.1-70b',
            $$
            {prompt}
            $$
        )
    """).collect()[0][0]

    sql = sql.replace("```sql","").replace("```","").strip()

    return sql


# -----------------------------------------------------------------------------
# Execute SQL
# -----------------------------------------------------------------------------
def run_query(question):

    try:

        sql = generate_sql(question)

        df = session.sql(sql).to_pandas()

        return df

    except Exception as e:
        return pd.DataFrame({"Error":[str(e)]})


# -----------------------------------------------------------------------------
# PII / PHI Masking
# -----------------------------------------------------------------------------
def mask_sensitive_data(df):

    pii_keywords = [
        "name",
        "first",
        "last",
        "email",
        "phone",
        "mobile",
        "ssn",
        "dob",
        "address"
    ]

    for col in df.columns:

        if any(k in col.lower() for k in pii_keywords):

            df[col] = df[col].astype(str).str[:2] + "****"

    return df


# -----------------------------------------------------------------------------
# Convert Data to Natural Language
# -----------------------------------------------------------------------------
def generate_answer(question, df):

    data_text = df.to_string(index=False)

    prompt = f"""
User Question:
{question}

Database Result:
{data_text}

Explain the answer in natural language.
Avoid exposing PII or PHI.
"""

    answer = session.sql(f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
        'llama3.1-70b',
        $$
        {prompt}
        $$
        )
    """).collect()[0][0]

    return answer


# -----------------------------------------------------------------------------
# Chat Interface
# -----------------------------------------------------------------------------
question = st.chat_input("Ask about patients, treatments, reports...")

if question:

    with st.spinner("Searching database..."):

        df = run_query(question)

        df = mask_sensitive_data(df)

        answer = generate_answer(question, df)

        st.write(answer)

        st.dataframe(df)
