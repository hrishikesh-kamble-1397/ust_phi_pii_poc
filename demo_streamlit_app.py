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
# Settings (Optimized for Token Safety)
# -----------------------------------------------------------------------------
MODEL_NAME = "mistral-large2"
EMBED_MODEL = "snowflake-arctic-embed-m"

SIMILARITY_THRESHOLD = 0.40
MAX_CHUNKS = 8                 # Reduced for better precision
MAX_CONTEXT_CHARS = 12000      # Hard context cap (~3k–4k tokens safe)

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
def call_llm(prompt):
    sql = """
        SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER
    """
    result = session.sql(sql, params=[MODEL_NAME, prompt]).collect()
    return result[0]["ANSWER"]

# -----------------------------------------------------------------------------
# Masking
# -----------------------------------------------------------------------------
def mask_answer(answer_text):
    masking_prompt = f"""
You are a compliance assistant.

Mask ALL PII and PHI including:
- Names
- Dates of birth
- Addresses
- Phone numbers
- IDs
- Lab values tied to a person
- Any identifiable patient information

Replace each sensitive value with exactly: XXXXXX

Return only masked text.

Text:
{answer_text}
"""
    return call_llm(masking_prompt)

# -----------------------------------------------------------------------------
# Presigned URL
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
# Hybrid Search
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
        ),

        keyword_results AS (
            SELECT
                CHUNK_TEXT,
                SOURCE_FILE,
                PAGE_NUM,
                0.75 AS SCORE
            FROM AI_POC_DB.PII_PHI_POC.DOCS_CHUNKS_NEW
            WHERE CHUNK_TEXT ILIKE '%' || ? || '%'
        )

        SELECT *
        FROM (
            SELECT * FROM vector_results
            UNION ALL
            SELECT * FROM keyword_results
        )
        WHERE SCORE >= {SIMILARITY_THRESHOLD}
        ORDER BY SCORE DESC
        LIMIT {MAX_CHUNKS}
    """

    return session.sql(search_sql, params=[query, query]).to_pandas()

# -----------------------------------------------------------------------------
# LOGIN SCREEN
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:

    st.title("🔐 Chatbot Login")

    with st.form("login_form"):
        login_user = st.text_input("Username", placeholder="e.g. Vedant")
        login_password = st.text_input("Password", type="password")
        login_btn = st.form_submit_button("Login")

    if login_btn:
        role = authenticate_user(login_user, login_password)

        if not role:
            st.error("Invalid credentials")
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
# MAIN APPLICATION
# -----------------------------------------------------------------------------
st.title("📄 PDF Chatbot on Snowflake")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask about your PDFs")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            try:
                chunks_df = call_search(prompt)

                if chunks_df.empty:
                    answer = "Information not found in documents."
                    st.write(answer)

                else:

                    # ---------------------------------------------------------
                    # SAFE CONTEXT BUILDER (12K CHAR CAP)
                    # ---------------------------------------------------------
                    context_parts = []
                    current_length = 0

                    for _, row in chunks_df.iterrows():

                        chunk_block = (
                            f"[File: {row.SOURCE_FILE} | Page: {row.PAGE_NUM}]\n"
                            f"{row.CHUNK_TEXT}\n\n"
                        )

                        if current_length + len(chunk_block) > MAX_CONTEXT_CHARS:
                            break

                        context_parts.append(chunk_block)
                        current_length += len(chunk_block)

                    context_text = "".join(context_parts)

                    full_prompt = f"""
You are a medical document assistant.

STRICT RULES:
- Use ONLY the provided context.
- If answer not explicitly present, say:
  "Information not found in documents."
- Do NOT infer missing names.
- Do NOT guess.
- Preserve numeric values exactly.

Context:
{context_text}

Question:
{prompt}

Answer:
"""

                    answer = call_llm(full_prompt)

                    if st.session_state.app_role not in ["admin", "owner"]:
                        answer = mask_answer(answer)

                    st.write(answer)

                    # ---------------------------------------------------------
                    # Most Relevant File Download
                    # ---------------------------------------------------------
                    file_scores = (
                        chunks_df
                        .groupby("SOURCE_FILE")["SCORE"]
                        .max()
                        .reset_index()
                        .sort_values("SCORE", ascending=False)
                    )

                    best_file = file_scores.iloc[0]["SOURCE_FILE"]
                    best_score = file_scores.iloc[0]["SCORE"]

                    if best_score >= SIMILARITY_THRESHOLD:
                        st.markdown("### 📥 Most Relevant PDF")
                        url = get_presigned_url(best_file)
                        st.link_button(
                            f"Download {best_file} (Score: {best_score:.3f})",
                            url
                        )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
