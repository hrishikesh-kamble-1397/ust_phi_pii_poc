import streamlit as st
import pandas as pd
import re
from snowflake.snowpark.context import get_active_session

# -----------------------------------------------------------------------------
# Streamlit Config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Database Chatbot", layout="wide")
st.title("🤖 AI Database Chatbot (Auto Schema Discovery)")

session = get_active_session()

# -----------------------------------------------------------------------------
# Get Database Schema Automatically
# -----------------------------------------------------------------------------
@st.cache_data
def get_schema():

    query = """
    SELECT TABLE_NAME, COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    ORDER BY TABLE_NAME
    """

    df = session.sql(query).to_pandas()

    schema_text=""

    for table in df["TABLE_NAME"].unique():

        cols=df[df["TABLE_NAME"]==table]["COLUMN_NAME"].tolist()

        schema_text+=f"{table}({','.join(cols)})\n"

    return schema_text


schema_info = get_schema()


# -----------------------------------------------------------------------------
# Mask PII / PHI
# -----------------------------------------------------------------------------
def mask_sensitive_data(text):

    if text is None:
        return ""

    text = re.sub(r'\S+@\S+', '[EMAIL_MASKED]', text)
    text = re.sub(r'\b\d{10}\b', '[PHONE_MASKED]', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_MASKED]', text)

    return text


# -----------------------------------------------------------------------------
# Generate SQL dynamically
# -----------------------------------------------------------------------------
def generate_sql(question):

    prompt = f"""
You are an expert Snowflake SQL developer.

You MUST use only the tables and columns listed below.

DATABASE SCHEMA:
{schema_info}

Instructions:
1. Only use table names and column names from the schema above.
2. If the user's words do not exactly match a column, choose the closest meaning column.
3. If multiple tables exist, create appropriate joins.
4. Do NOT invent columns.
5. Return ONLY Snowflake SQL.
6. Query must start with SELECT.

User Question:
{question}
"""

    result = session.sql(f"""
    SELECT SNOWFLAKE.CORTEX.COMPLETE(
        'llama3.1-70b',
        $$ {prompt} $$
    )
    """).collect()

    sql = result[0][0]

    if sql is None:
        return "SELECT 'Unable to generate SQL'"

    sql = sql.replace("```sql","").replace("```","").strip()

    return sql

# -----------------------------------------------------------------------------
# Validate SQL
# -----------------------------------------------------------------------------
def validate_sql(sql):

    sql_upper = sql.upper()

    forbidden = ["DROP","DELETE","UPDATE","INSERT","ALTER"]

    for word in forbidden:
        if word in sql_upper:
            return False

    return True


# -----------------------------------------------------------------------------
# Execute Query
# -----------------------------------------------------------------------------
def run_query(sql, question):

    try:
        return session.sql(sql).to_pandas()

    except Exception as e:

        error=str(e)

        fix_prompt=f"""
The following Snowflake SQL failed.

SQL:
{sql}

Error:
{error}

Database schema:
{schema_info}

Fix the SQL.
Return only corrected SQL.
"""

        fixed_sql=session.sql(f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
        'llama3.1-70b',
        $$ {fix_prompt} $$
        )
        """).collect()[0][0]

        fixed_sql=fixed_sql.replace("```","").strip()

        return session.sql(fixed_sql).to_pandas()
# -----------------------------------------------------------------------------
# Generate Natural Language Response
# -----------------------------------------------------------------------------
def generate_answer(question, result):

    prompt = f"""
User question:
{question}

Query result:
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
# Chat History
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# -----------------------------------------------------------------------------
# Chat Input
# -----------------------------------------------------------------------------
user_question = st.chat_input("Ask anything about your database")

if user_question:

    st.chat_message("user").write(user_question)

    st.session_state.messages.append({
        "role":"user",
        "content":user_question
    })

    with st.spinner("Analyzing database..."):

        sql_query = generate_sql(user_question)

        st.code(sql_query, language="sql")

        if validate_sql(sql_query):

            df = run_query(sql_query)

            result_text = df.to_string()

            result_text = mask_sensitive_data(result_text)

            answer = generate_answer(user_question, result_text)

        else:
            answer = "Query blocked due to security restrictions."

    st.chat_message("assistant").write(answer)

    st.session_state.messages.append({
        "role":"assistant",
        "content":answer
    })
