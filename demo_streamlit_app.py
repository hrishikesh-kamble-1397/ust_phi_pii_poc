import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

# -----------------------------------------------------------------------------
# APP CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

session = get_active_session()

# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------
MODEL_NAME = "llama3.1-70b"
EMBED_MODEL = "snowflake-arctic-embed-m"

SIMILARITY_THRESHOLD = 0.65
TOP_FILES = 5
CHUNKS_PER_FILE = 3
MAX_CONTEXT_CHUNKS = TOP_FILES * CHUNKS_PER_FILE

STAGE_NAME = "AI_POC_DB.PII_PHI_POC.PHI_PII_POC_STAGE1"

# -----------------------------------------------------------------------------
# SESSION STATE
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
# AUTHENTICATION
# -----------------------------------------------------------------------------
def authenticate_user(user_name, password):

    df = session.sql("""
        SELECT APP_ROLE
        FROM AI_POC_DB.PII_PHI_POC.APP_USER_ACCESS
        WHERE (
            UPPER(USER_NAME) = UPPER(:1)
            OR UPPER(USER_NAME) = SPLIT(UPPER(:1),'@')[0]
        )
        AND PASSWORD = :2
        AND IS_ACTIVE = TRUE
    """, [user_name, password]).to_pandas()

    if df.empty:
        return None

    return df.iloc[0]["APP_ROLE"].lower()

# -----------------------------------------------------------------------------
# LLM CALL
# -----------------------------------------------------------------------------
def call_llm(prompt):

    sql = """
    SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS RESPONSE
    """

    row = session.sql(sql, params=[MODEL_NAME, prompt]).collect()[0]

    return row["RESPONSE"]

# -----------------------------------------------------------------------------
# QUERY REWRITE
# -----------------------------------------------------------------------------
def rewrite_query(question):

    prompt = f"""
Rewrite the question into a concise semantic search query
optimized for retrieving medical document content.

Return only the rewritten query.

Question:
{question}

Search Query:
"""

    return call_llm(prompt).strip()

# -----------------------------------------------------------------------------
# STAGE 1 RETRIEVAL - TOP FILES
# -----------------------------------------------------------------------------
def retrieve_top_files(query):

    sql = f"""
    WITH query_vec AS (
        SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('{EMBED_MODEL}', ?) emb
    )
    SELECT
        SOURCE_FILE,
        MAX(VECTOR_COSINE_SIMILARITY(EMBEDDING, q.emb)) SCORE
    FROM AI_POC_DB.PII_PHI_POC.CLEANED_CHUNKS c
    CROSS JOIN query_vec q
    GROUP BY SOURCE_FILE
    ORDER BY SCORE DESC
    LIMIT {TOP_FILES}
    """

    return session.sql(sql, params=[query]).to_pandas()

# -----------------------------------------------------------------------------
# STAGE 2 RETRIEVAL - CHUNKS
# -----------------------------------------------------------------------------
def retrieve_chunks(query, files):

    file_list = ",".join([f"'{f}'" for f in files])

    sql = f"""
    WITH query_vec AS (
        SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768('{EMBED_MODEL}', ?) emb
    ),
    ranked_chunks AS (
        SELECT
            CLEANED_CHUNK_TEXT,
            SOURCE_FILE,
            VECTOR_COSINE_SIMILARITY(EMBEDDING, q.emb) SCORE,
            ROW_NUMBER() OVER(
                PARTITION BY SOURCE_FILE
                ORDER BY VECTOR_COSINE_SIMILARITY(EMBEDDING, q.emb) DESC
            ) RN
        FROM AI_POC_DB.PII_PHI_POC.CLEANED_CHUNKS c
        CROSS JOIN query_vec q
        WHERE SOURCE_FILE IN ({file_list})
    )

    SELECT *
    FROM ranked_chunks
    WHERE RN <= {CHUNKS_PER_FILE}
    AND SCORE >= {SIMILARITY_THRESHOLD}
    ORDER BY SCORE DESC
    LIMIT {MAX_CONTEXT_CHUNKS}
    """

    return session.sql(sql, params=[query]).to_pandas()

# -----------------------------------------------------------------------------
# ANSWER GENERATION
# -----------------------------------------------------------------------------
def generate_answer(question, context):

    prompt = f"""
You are a medical document assistant.

RULES:
- Answer ONLY using the provided context
- If answer not found say:
"The documents do not contain this information."
- Be concise and factual

Context:
{context}

Question:
{question}

Answer:
"""

    return call_llm(prompt)

# -----------------------------------------------------------------------------
# MASKING
# -----------------------------------------------------------------------------
def mask_answer(answer):

    prompt = f"""
Mask all PII and PHI in the text below.
Replace sensitive values with exactly "XXXXXX".

Return only masked text.

Text:
{answer}
"""

    return call_llm(prompt)

# -----------------------------------------------------------------------------
# PRESIGNED URL
# -----------------------------------------------------------------------------
def get_presigned_url(file):

    sql = f"""
    SELECT GET_PRESIGNED_URL(
        @{STAGE_NAME},
        '{file}',
        3600
    ) URL
    """

    return session.sql(sql).collect()[0]["URL"]

# -----------------------------------------------------------------------------
# LOGIN SCREEN
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:

    st.title("🔐 Chatbot Login")

    with st.form("login_form"):

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        login = st.form_submit_button("Login")

    if login:

        role = authenticate_user(username, password)

        if not role:
            st.error("Invalid credentials")
            st.stop()

        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.app_role = role

        st.rerun()

    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR
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
st.title("📄 Snowflake PDF Chatbot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask about your PDFs")

# -----------------------------------------------------------------------------
# CHAT FLOW
# -----------------------------------------------------------------------------
if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            try:

                # Step 1 Rewrite query
                search_query = rewrite_query(prompt)

                # Step 2 Find relevant files
                files_df = retrieve_top_files(search_query)

                if files_df.empty:

                    answer = "No relevant documents found."

                    st.write(answer)

                else:

                    files = files_df["SOURCE_FILE"].tolist()

                    # Step 3 Retrieve chunks
                    chunks_df = retrieve_chunks(search_query, files)

                    if chunks_df.empty:

                        answer = "No relevant document content found."

                        st.write(answer)

                    else:

                        # Build context
                        context = "\n\n".join(
                            f"Document: {row.SOURCE_FILE}\nText: {row.CLEANED_CHUNK_TEXT}"
                            for _, row in chunks_df.iterrows()
                        )

                        # Generate answer
                        answer = generate_answer(prompt, context)

                        # Mask if not admin
                        if st.session_state.app_role not in ["admin","owner"]:
                            answer = mask_answer(answer)

                        st.write(answer)

                        # Show best PDF
                        best_file = files_df.iloc[0]["SOURCE_FILE"]

                        url = get_presigned_url(best_file)

                        st.markdown("### 📥 Most Relevant PDF")

                        st.link_button(f"Download {best_file}", url)

                st.session_state.messages.append(
                    {"role":"assistant","content":answer}
                )

            except Exception as e:

                error = f"Error: {str(e)}"

                st.error(error)

                st.session_state.messages.append(
                    {"role":"assistant","content":error}
                )
