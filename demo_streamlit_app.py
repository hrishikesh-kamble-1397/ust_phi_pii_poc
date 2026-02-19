import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

# -----------------------------------------------------------------------------
# App Config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

# -----------------------------------------------------------------------------
# Snowflake Session
# -----------------------------------------------------------------------------
session = get_active_session()
STAGE_NAME = "AI_POC_DB.PII_PHI_POC.PHI_PII_POC_STAGE1"

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

if "app_role" not in st.session_state:
    st.session_state.app_role = None

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your PDFs."}
    ]

# -----------------------------------------------------------------------------
# Authentication
# -----------------------------------------------------------------------------
def authenticate_user(user_name, password):

    df = session.sql("""
        SELECT APP_ROLE
        FROM AI_POC_DB.PII_PHI_POC.APP_USER_ACCESS
        WHERE (
            UPPER(USER_NAME) = UPPER(:1)
            OR UPPER(USER_NAME) = SPLIT(UPPER(:1), '@')[0]
        )
        AND PASSWORD = :2
        AND IS_ACTIVE = TRUE
    """, [user_name, password]).to_pandas()

    if df.empty:
        return None

    return df.iloc[0]["APP_ROLE"].lower()

# -----------------------------------------------------------------------------
# LLM Call
# -----------------------------------------------------------------------------
def call_llm(model_name, prompt):
    sql = "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER"
    row = session.sql(sql, params=[model_name, prompt]).collect()[0]
    return row["ANSWER"]

# -----------------------------------------------------------------------------
# Mask Answer for Non-Admin
# -----------------------------------------------------------------------------
def mask_answer_with_llm(answer_text):

    masking_prompt = f"""
Mask ALL PII and PHI in the text below.
Replace sensitive values with exactly "XXXXXX".
Return only masked text.

Text:
{answer_text}
"""

    return call_llm("llama3.1-70b", masking_prompt)

# -----------------------------------------------------------------------------
# Generate Presigned URL
# -----------------------------------------------------------------------------
def get_presigned_url(file_name):

    sql = f"""
        SELECT GET_PRESIGNED_URL(
            @{STAGE_NAME},
            '{file_name}',
            3600
        ) AS URL
    """

    return session.sql(sql).collect()[0]["URL"]

# -----------------------------------------------------------------------------
# Vector Search
# -----------------------------------------------------------------------------
def call_search(query, k):

    search_sql = f"""
        WITH query_vec AS (
            SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
                'snowflake-arctic-embed-m',
                ?
            ) AS emb
        )
        SELECT
            c.CHUNK_TEXT,
            c.SOURCE_FILE,
            VECTOR_COSINE_SIMILARITY(c.EMBEDDING, q.emb) AS SCORE
        FROM DOCS_CHUNKS c
        CROSS JOIN query_vec q
        ORDER BY SCORE DESC
        LIMIT {k}
    """

    return session.sql(search_sql, params=[query]).to_pandas()

# -----------------------------------------------------------------------------
# Fetch Distinct Entity Names
# -----------------------------------------------------------------------------
def fetch_distinct_entities(column_name):

    sql = f"""
        SELECT DISTINCT {column_name}
        FROM DOCUMENT_METADATA_HISTORY
        WHERE {column_name} IS NOT NULL
        ORDER BY {column_name}
    """

    return session.sql(sql).to_pandas()[column_name].tolist()

# -----------------------------------------------------------------------------
# LOGIN
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:

    st.title("🔐 Chatbot Login")

    with st.form("login_form"):
        login_user = st.text_input("Username")
        login_password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

    if login_btn:
        role = authenticate_user(login_user, login_password)

        if not role:
            st.error("Invalid username or password.")
            st.stop()

        st.session_state.authenticated = True
        st.session_state.username = login_user
        st.session_state.app_role = role
        st.rerun()

    st.stop()

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.success("Authenticated")
st.sidebar.write("User:", st.session_state.username)
st.sidebar.write("Role:", st.session_state.app_role.upper())

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# -----------------------------------------------------------------------------
# ADMIN / OWNER TABS
# -----------------------------------------------------------------------------
if st.session_state.app_role in ["admin", "owner"]:

    st.sidebar.markdown("## 📂 Data Explorer")

    tab = st.sidebar.radio(
        "Select Category",
        ["Patient Details", "Doctor Details", "Hospital Details"]
    )

    search_text = st.sidebar.text_input("Search")

    if tab == "Patient Details":
        names = fetch_distinct_entities("PATIENT_NAME")

    elif tab == "Doctor Details":
        names = fetch_distinct_entities("DOCTOR_NAME")

    else:
        names = fetch_distinct_entities("HOSPITAL_NAME")

    if search_text:
        names = [n for n in names if search_text.lower() in n.lower()]

    selected = st.sidebar.selectbox("Select Name", names)

    if st.sidebar.button("View Details"):
        query = f"Provide complete details about {selected}"

        st.session_state.messages.append({"role": "user", "content": query})
        st.rerun()

# -----------------------------------------------------------------------------
# Main Chat Window
# -----------------------------------------------------------------------------
st.title("📄 PDF Chatbot on Snowflake")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Type your question")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                chunks_df = call_search(prompt, 10)

                if chunks_df.empty:
                    answer = "No relevant content found."
                    st.write(answer)

                else:
                    context_text = "\n\n---\n\n".join(
                        chunks_df["CHUNK_TEXT"].tolist()
                    )

                    full_prompt = f"""
Answer using ONLY the context below.
Be precise.

Context:
{context_text}

Question:
{prompt}
Answer:
"""

                    answer = call_llm("llama3.1-70b", full_prompt)

                    if st.session_state.app_role not in ["admin", "owner"]:
                        answer = mask_answer_with_llm(answer)

                    st.write(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error(str(e))
