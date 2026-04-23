import streamlit as st
import pandas as pd
import os
from snowflake.snowpark.context import get_active_session
from anthropic import Anthropic

# -----------------------------------------------------------------------------
# App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PDF/Database Chatbot", page_icon="📄", layout="wide")

# -----------------------------------------------------------------------------
# Snowflake Session
# -----------------------------------------------------------------------------
session = get_active_session()

# -----------------------------------------------------------------------------
# Claude Setup
# -----------------------------------------------------------------------------
claude_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
CLAUDE_MODEL = "claude-3-7-sonnet-20250219"

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
MODEL_NAME = "mistral-large2"
EMBED_MODEL = "snowflake-arctic-embed-m"
SIMILARITY_THRESHOLD = 0.35
MAX_CHUNKS = 50

# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

if "app_role" not in st.session_state:
    st.session_state.app_role = None

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your PDFs or database."}
    ]

if "mode" not in st.session_state:
    st.session_state.mode = "PDF"

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
# Cortex LLM (used for PDF + masking)
# -----------------------------------------------------------------------------
def call_llm(prompt):
    sql = """ SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER """
    result = session.sql(sql, params=[MODEL_NAME, prompt]).collect()
    return result[0]["ANSWER"]

# -----------------------------------------------------------------------------
# Claude Call
# -----------------------------------------------------------------------------
def call_claude(prompt):
    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=800,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# -----------------------------------------------------------------------------
# Masking (for non-admin)
# -----------------------------------------------------------------------------
def mask_answer(answer_text):
    masking_prompt = f"""
Mask ALL PII and PHI.
Replace sensitive values with: XXXXXX
Return only masked text.
Do NOT add new information.

Text:
{answer_text}
"""
    return call_llm(masking_prompt)

# -----------------------------------------------------------------------------
# PDF Functions (UNCHANGED)
# -----------------------------------------------------------------------------
def call_search(query):
    search_sql = f"""
        WITH query_vec AS (
            SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
                '{EMBED_MODEL}',
                ?
            ) AS emb
        ),
        vector_results AS (
            SELECT
                c.CHUNK_TEXT,
                c.SOURCE_FILE,
                c.PAGE_NUM,
                VECTOR_COSINE_SIMILARITY(c.EMBEDDING, q.emb) AS SCORE
            FROM AI_POC_DB.PII_PHI_POC.DOCS_CHUNKS_NEW c
            CROSS JOIN query_vec q
            WHERE c.EMBEDDING IS NOT NULL
        )
        SELECT *
        FROM vector_results
        WHERE SCORE >= {SIMILARITY_THRESHOLD}
        ORDER BY SCORE DESC
        LIMIT {MAX_CHUNKS}
    """
    return session.sql(search_sql, params=[query]).to_pandas()

# -----------------------------------------------------------------------------
# DATABASE FUNCTIONS (FIXED + BATCH OPTIMIZED)
# -----------------------------------------------------------------------------
def get_relevant_batches(user_prompt):
    notes_col = "EHR_NOTES" if st.session_state.app_role in ["admin", "owner"] else "NOTES_REDACTED"

    sql = f"""
        SELECT BATCH_ID, COUNT(*) AS CNT
        FROM AI_POC_DB.PII_PHI_POC.POC_EHR_NOTES_PHI_REDACTED_OPT
        WHERE SEARCH(({notes_col}, HP_DETAILS), :1, SEARCH_MODE => 'OR')
        GROUP BY BATCH_ID
        ORDER BY CNT DESC
        LIMIT 5
    """

    return session.sql(sql, params=[user_prompt]).to_pandas()


def get_batch_data(batch_ids):
    notes_col = "EHR_NOTES" if st.session_state.app_role in ["admin", "owner"] else "NOTES_REDACTED"

    batch_list = ",".join([str(x) for x in batch_ids])

    sql = f"""
        SELECT
            BATCH_ID,
            LISTAGG(
                'PatientID: ' || PATIENT_ID ||
                ' | Name: ' || PATIENT_NAME ||
                ' | Details: ' || HP_DETAILS ||
                ' | Notes: ' || {notes_col},
                '\n---\n'
            ) AS BATCH_TEXT
        FROM AI_POC_DB.PII_PHI_POC.POC_EHR_NOTES_PHI_REDACTED_OPT
        WHERE BATCH_ID IN ({batch_list})
        GROUP BY BATCH_ID
    """

    return session.sql(sql).to_pandas()


def build_prompt(user_prompt, batch_df):
    context = ""
    for _, row in batch_df.iterrows():
        context += f"\n===== BATCH {row['BATCH_ID']} =====\n{row['BATCH_TEXT']}\n"

    return f"""
You are a clinical data assistant.

Use ONLY context below.
If answer not found, reply exactly:
"Information not found in database."

Context:
{context}

Question:
{user_prompt}

Answer:
"""


def get_db_answer(user_prompt):
    batch_df = get_relevant_batches(user_prompt)

    if batch_df.empty:
        return "Information not found in database."

    batch_ids = batch_df["BATCH_ID"].tolist()
    data_df = get_batch_data(batch_ids)

    if data_df.empty:
        return "Information not found in database."

    prompt = build_prompt(user_prompt, data_df)

    answer = call_claude(prompt)

    if st.session_state.app_role not in ["admin", "owner"]:
        answer = mask_answer(answer)

    return answer

# -----------------------------------------------------------------------------
# LOGIN
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:
    st.title("🔐 Chatbot Login")

    with st.form("login_form"):
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        btn = st.form_submit_button("Login")

    if btn:
        role = authenticate_user(user, pwd)
        if not role:
            st.error("Invalid credentials")
            st.stop()

        st.session_state.authenticated = True
        st.session_state.username = user
        st.session_state.app_role = role
        st.rerun()

    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
st.sidebar.success("Authenticated")
st.sidebar.write("User:", st.session_state.username)
st.sidebar.write("Role:", st.session_state.app_role.upper())

st.session_state.mode = st.sidebar.radio("Mode", ["PDF", "Database"])

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# -----------------------------------------------------------------------------
# CHAT UI
# -----------------------------------------------------------------------------
st.title("📄 PDF / Database Chatbot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask something")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                if st.session_state.mode == "PDF":
                    chunks_df = call_search(prompt)

                    if chunks_df.empty:
                        answer = "Information not found in documents."
                    else:
                        context = "\n\n".join(
                            [row.CHUNK_TEXT for _, row in chunks_df.iterrows()]
                        )

                        answer = call_llm(f"""
Use ONLY context below.

Context:
{context}

Question:
{prompt}

Answer:
""")

                        if st.session_state.app_role not in ["admin", "owner"]:
                            answer = mask_answer(answer)

                else:
                    answer = get_db_answer(prompt)

                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(str(e))
