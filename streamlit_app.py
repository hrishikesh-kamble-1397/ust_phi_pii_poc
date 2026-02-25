import streamlit as st
import pandas as pd
import re
from snowflake.snowpark.context import get_active_session

# -----------------------------------------------------------------------------
# APP CONFIG
# -----------------------------------------------------------------------------

st.set_page_config(page_title="AI Healthcare Data Chatbot", layout="wide")
st.title("🤖 AI Database Chatbot")

session = get_active_session()

# -----------------------------------------------------------------------------
# FETCH ALL TABLES
# -----------------------------------------------------------------------------

@st.cache_data
def get_all_tables():

    query = """
    SELECT table_name
    FROM AI_POC_DB.information_schema.tables
    WHERE table_schema = CURRENT_SCHEMA()
    """

    return session.sql(query).to_pandas()

# -----------------------------------------------------------------------------
# IDENTIFY RELEVANT TABLES USING AI
# -----------------------------------------------------------------------------

def identify_tables(question):

    tables = get_all_tables()

    table_list = ",".join(tables["TABLE_NAME"].tolist())

    prompt = f"""
You are a Snowflake database expert.

Available tables:
{table_list}

User Question:
{question}

Identify the tables needed to answer the question.

Return comma separated table names only.
"""

    result = session.sql(f"""
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'llama3.1-70b',
    $$ {prompt} $$
    )
    """).collect()[0][0]

    tables = [t.strip() for t in result.split(",")]

    return tables

# -----------------------------------------------------------------------------
# GET SCHEMA FOR SELECTED TABLES
# -----------------------------------------------------------------------------

def get_schema(tables):

    table_string = ",".join([f"'{t}'" for t in tables])

    query = f"""
    SELECT table_name,column_name
    FROM information_schema.columns
    WHERE table_name IN ({table_string})
    """

    df = session.sql(query).to_pandas()

    schema = ""

    for table in df["TABLE_NAME"].unique():

        cols = df[df["TABLE_NAME"] == table]["COLUMN_NAME"].tolist()

        schema += f"""
Table: {table}
Columns: {','.join(cols)}
"""

    return schema

# -----------------------------------------------------------------------------
# SQL GENERATION
# -----------------------------------------------------------------------------

def generate_sql(question, schema):

    prompt = f"""
You are a Snowflake SQL expert.

Database Schema:
{schema}

User Question:
{question}

Rules:

1 Use joins if required
2 Detect patient name fields automatically
   (first_name,last_name,full_name,patient_name)
3 Return SQL only
4 Limit results to 50 rows

SQL:
"""

    sql = session.sql(f"""
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'llama3.1-70b',
    $$ {prompt} $$
    )
    """).collect()[0][0]

    sql = sql.replace("```sql","").replace("```","").strip()

    return sql

# -----------------------------------------------------------------------------
# SQL SAFETY
# -----------------------------------------------------------------------------

def validate_sql(sql):

    forbidden = ["DROP","DELETE","UPDATE","INSERT","ALTER"]

    for word in forbidden:

        if word in sql.upper():

            return False

    return True

# -----------------------------------------------------------------------------
# EXECUTE QUERY
# -----------------------------------------------------------------------------

def run_query(sql):

    try:

        df = session.sql(sql).to_pandas()

        return df

    except Exception as e:

        return pd.DataFrame({"ERROR":[str(e)]})

# -----------------------------------------------------------------------------
# PII / PHI MASKING
# -----------------------------------------------------------------------------

def mask_phi(df):

    sensitive = [
        "name",
        "email",
        "phone",
        "ssn",
        "address",
        "dob"
    ]

    for col in df.columns:

        if any(word in col.lower() for word in sensitive):

            df[col] = df[col].astype(str).str[:2] + "****"

    return df

# -----------------------------------------------------------------------------
# NATURAL LANGUAGE RESPONSE
# -----------------------------------------------------------------------------

def generate_answer(question, df):

    result = df.to_string(index=False)

    prompt = f"""
User Question:
{question}

Database Result:
{result}

Explain the result clearly in natural language.
"""

    answer = session.sql(f"""
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'llama3.1-70b',
    $$ {prompt} $$
    )
    """).collect()[0][0]

    return answer

# -----------------------------------------------------------------------------
# CHAT STATE
# -----------------------------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []

for msg in st.session_state.messages:

    st.chat_message(msg["role"]).write(msg["content"])

# -----------------------------------------------------------------------------
# CHAT INPUT
# -----------------------------------------------------------------------------

question = st.chat_input("Ask about your healthcare database")

if question:

    st.chat_message("user").write(question)

    st.session_state.messages.append({
        "role":"user",
        "content":question
    })

    with st.spinner("Analyzing database..."):

        tables = identify_tables(question)

        schema = get_schema(tables)

        sql = generate_sql(question, schema)

        if validate_sql(sql):

            df = run_query(sql)

            df = mask_phi(df)

            answer = generate_answer(question, df)

        else:

            answer = "Query blocked due to security policy."

    st.chat_message("assistant").write(answer)

    st.session_state.messages.append({
        "role":"assistant",
        "content":answer
    })
