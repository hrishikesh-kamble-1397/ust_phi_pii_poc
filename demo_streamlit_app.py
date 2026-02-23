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
# Settings
# -----------------------------------------------------------------------------
MODEL_NAME = "llama3.1-70b"
EMBED_MODEL = "snowflake-arctic-embed-m"
SIMILARITY_THRESHOLD = 0.65
MAX_CHUNKS = 15   # Increased from 10

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
# LLM
# -----------------------------------------------------------------------------
def call_llm(prompt):
    sql = """
        SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER
    """
    row = session.sql(sql, params=[MODEL_NAME, prompt]).collect()[0]
    return row["ANSWER"]

# -----------------------------------------------------------------------------
# Multi-Prompt Step 1: Query Rewrite
# -----------------------------------------------------------------------------
def rewrite_query(user_question):
    rewrite_prompt = f"""
Rewrite the question into a concise search query
optimized for retrieving medical document content.

Return only the rewritten query.

Question:
{user_question}

Optimized Query:
"""
    return call_llm(rewrite_prompt).strip()

# -----------------------------------------------------------------------------
# Multi-Prompt Step 2: Grounded Answer
# -----------------------------------------------------------------------------
def generate_answer(question, context_text):
    answer_prompt = f"""
You are a medical document assistant.

RULES:
- Answer ONLY using the provided context.
- If the answer is not found in the context, say:
  "The documents do not contain this information."
- Do NOT use outside knowledge.
- Be concise and factual.

Context:
{context_text}

Question:
{question}

Answer:
"""
    return call_llm(answer_prompt)

# -----------------------------------------------------------------------------
# Multi-Prompt Step 3: Masking (PII/PHI Protection)
# -----------------------------------------------------------------------------
def mask_answer(answer_text):
    masking_prompt = f"""
Mask ALL PII and PHI in the text below.
Replace sensitive values with exactly "XXXXXX".
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
# VECTOR SEARCH (FIXED)
# -----------------------------------------------------------------------------
def call_search(query):
    search_sql = f"""
        WITH query_vec AS (
            SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
                '{EMBED_MODEL}',
                ?
            ) AS emb
        )
        SELECT *
        FROM (
            SELECT
                c.CLEANED_CHUNK_TEXT,
                c.SOURCE_FILE,
                VECTOR_COSINE_SIMILARITY(c.EMBEDDING, q.emb) AS SCORE
            FROM AI_POC_DB.PII_PHI_POC.CLEANED_CHUNKS c
            CROSS JOIN query_vec q
        )
        WHERE SCORE >= {SIMILARITY_THRESHOLD}
        ORDER BY SCORE DESC
        LIMIT {MAX_CHUNKS}
    """

    return session.sql(search_sql, params=[query]).to_pandas()
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
# MAIN APP
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
        with st.spinner("Processing..."):

            try:
                # STEP 1 — Query Rewrite
                optimized_query = rewrite_query(prompt)

                # STEP 2 — Vector Search
                chunks_df = call_search(optimized_query)

                if chunks_df.empty:
                    answer = "No relevant content found in documents."
                    st.write(answer)

                else:
                    context_text = "\n\n---\n\n".join(
                        chunks_df["CHUNK_TEXT"].tolist()
                    )

                    # STEP 3 — Grounded Answer
                    answer = generate_answer(prompt, context_text)

                    # STEP 4 — Role-Based Masking
                    if st.session_state.app_role not in ["admin", "owner"]:
                        answer = mask_answer(answer)

                    st.write(answer)

                    # Best Matching PDF
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
