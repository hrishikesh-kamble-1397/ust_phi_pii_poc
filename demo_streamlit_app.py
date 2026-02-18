import re
import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session
from datetime import datetime

# -----------------------------------------------------------------------------
# App config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

# -----------------------------------------------------------------------------
# Snowflake Session
# -----------------------------------------------------------------------------
session = get_active_session()

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

if "app_role" not in st.session_state:
    st.session_state.app_role = None

# -----------------------------------------------------------------------------
# Fetch App Role from DB
# -----------------------------------------------------------------------------
def get_app_role(user_name):

    df = session.sql("""
        SELECT APP_ROLE
        FROM AI_POC_DB.PII_PHI_POC.APP_USER_ACCESS
        WHERE (
            UPPER(USER_NAME) = UPPER(:1)
            OR UPPER(USER_NAME) = SPLIT(UPPER(:1), '@')[0]
        )
        AND IS_ACTIVE = TRUE
    """, [user_name]).to_pandas()

    if df.empty:
        return None

    return df.iloc[0]["APP_ROLE"].lower()

# -----------------------------------------------------------------------------
# LOGIN SCREEN
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:

    st.title("🔐 Chatbot Login")
    st.caption("Authenticate to access PDF Chatbot")

    with st.form("login_form"):
        login_user = st.text_input(
            "Username",
            placeholder="e.g. username or username@company.com"
        )
        login_btn = st.form_submit_button("Login")

    if login_btn:

        if not login_user.strip():
            st.warning("Please enter your username.")
            st.stop()

        role = get_app_role(login_user)

        if not role:
            st.error("❌ You are not authorized to access this application.")
            st.stop()

        st.session_state.authenticated = True
        st.session_state.username = login_user
        st.session_state.app_role = role

        st.rerun()

    st.stop()

# -----------------------------------------------------------------------------
# Sidebar – User Info
# -----------------------------------------------------------------------------
st.sidebar.success("Authenticated")
st.sidebar.write("👤 User:", st.session_state.username)
st.sidebar.write("🛡️ App Role:", st.session_state.app_role.upper())

if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
st.title("📄 PDF Chatbot on Snowflake")

# -----------------------------------------------------------------------------
# Sidebar Settings
# -----------------------------------------------------------------------------
st.sidebar.header("Settings")

model = st.sidebar.selectbox(
    "LLM model",
    options=[
        "mistral-large",
        "llama3.1-70b",
        "mixtral-8x7b"
    ],
    index=0
)

top_k = st.sidebar.slider("Top-K chunks", 1, 10, 5)

# Only admin can toggle source visibility
show_sources = False
if st.session_state.app_role == "admin":
    show_sources = st.sidebar.checkbox("Show sources", value=True)

# -----------------------------------------------------------------------------
# Admin-only PII/PHI Section
# -----------------------------------------------------------------------------
if st.session_state.app_role == "admin":

    st.sidebar.markdown("---")
    st.sidebar.subheader("PII/PHI Extractor")

    default_stage_file = "sample_hospitalization_claim1.pdf"

    stage_file_name = st.sidebar.text_input(
        "Stage file name",
        value=default_stage_file
    )

    run_proc = st.sidebar.button("Run PII/PHI Parse & Classify")

    if run_proc:
        with st.spinner("Running PII/PHI procedure..."):
            try:
                sql = "CALL AI_POC_DB.PII_PHI_POC.SP_PARSE_EXTRACT_CLASSIFY(?)"
                row = session.sql(sql, params=[stage_file_name]).collect()[0]
                result_msg = list(row.asDict().values())[0]

                st.sidebar.success(result_msg)

            except Exception as e:
                st.sidebar.error(f"Procedure failed: {e}")

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def call_search(query: str, k: int) -> pd.DataFrame:

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


def call_llm(model_name: str, prompt: str) -> str:

    llm_sql = """
        SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER
    """

    row = session.sql(llm_sql, params=[model_name, prompt]).collect()[0]
    return row["ANSWER"]

# -----------------------------------------------------------------------------
# Chat History
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your PDFs."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------------------------------------------------------
# Chat Input
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Type your question about the PDFs"):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                chunks_df = call_search(prompt, top_k)

                if chunks_df.empty:
                    answer = "No relevant content found in documents."
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    context_blocks = []

                    for _, row in chunks_df.iterrows():

                        txt = row["CHUNK_TEXT"]

                        # Mask content for non-admin users
                        if st.session_state.app_role != "admin":
                            txt = "[REDACTED CONTENT]"

                        context_blocks.append(
                            f"File: {row['SOURCE_FILE']}\n"
                            f"Score: {row['SCORE']:.4f}\n"
                            f"Content:\n{txt}\n"
                        )

                    context_text = "\n\n---\n\n".join(context_blocks)

                    system_prompt = (
                        "Answer strictly using provided context. "
                        "If answer not found, say you don't know."
                    )

                    full_prompt = f"""
{system_prompt}

Context:
{context_text}

Question: {prompt}
Answer:
"""

                    answer = call_llm(model, full_prompt)

                    st.write(answer)

                    # Admin-only source visibility
                    if show_sources:
                        with st.expander("Retrieved Sources"):
                            st.dataframe(
                                chunks_df[["SOURCE_FILE", "SCORE", "CHUNK_TEXT"]],
                                use_container_width=True
                            )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

            except Exception as e:
                err_msg = f"Error: {e}"
                st.error(err_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err_msg}
                )
