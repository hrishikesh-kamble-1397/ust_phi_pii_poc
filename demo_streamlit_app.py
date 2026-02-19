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

SIMILARITY_THRESHOLD = 0.65
MODEL_NAME = "llama3.1-70b"

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

    sql = """SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER"""
    row = session.sql(sql, params=[model_name, prompt]).collect()[0]
    return row["ANSWER"]

# -----------------------------------------------------------------------------
# Vector Search (Generic)
# -----------------------------------------------------------------------------
def vector_search(query, top_k=5):

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
        LIMIT {top_k}
    """

    return session.sql(search_sql, params=[query]).to_pandas()

# -----------------------------------------------------------------------------
# Validate Entity Has Context
# -----------------------------------------------------------------------------
def entity_has_context(name):

    search_sql = """
        WITH query_vec AS (
            SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
                'snowflake-arctic-embed-m',
                ?
            ) AS emb
        )
        SELECT
            MAX(VECTOR_COSINE_SIMILARITY(c.EMBEDDING, q.emb)) AS MAX_SCORE
        FROM DOCS_CHUNKS c
        CROSS JOIN query_vec q
    """

    result = session.sql(search_sql, params=[name]).collect()[0]
    max_score = result["MAX_SCORE"]

    if max_score and max_score >= SIMILARITY_THRESHOLD:
        return True
    return False

# -----------------------------------------------------------------------------
# Extract & Validate Entities
# -----------------------------------------------------------------------------
def extract_entities(entity_type):

    cache_key = f"{entity_type}_entities"

    if cache_key in st.session_state:
        return st.session_state[cache_key]

    all_chunks = session.sql(
        "SELECT CHUNK_TEXT FROM DOCS_CHUNKS"
    ).to_pandas()

    full_text = "\n\n".join(all_chunks["CHUNK_TEXT"].tolist())

    prompt = f"""
Extract ALL unique {entity_type} names from the text below.
Return ONLY comma-separated list.
No explanation.

Text:
{full_text}

Output:
"""

    response = call_llm(MODEL_NAME, prompt)

    raw_entities = [e.strip() for e in response.split(",") if e.strip()]

    valid_entities = []

    for name in raw_entities:
        if entity_has_context(name):
            valid_entities.append(name)

    valid_entities = sorted(list(set(valid_entities)))

    st.session_state[cache_key] = valid_entities

    return valid_entities

# -----------------------------------------------------------------------------
# Generate Detailed Answer For Entity
# -----------------------------------------------------------------------------
def generate_entity_answer(name, entity_type):

    chunks_df = vector_search(name, top_k=8)

    if chunks_df.empty:
        return f"No detailed information found for {name}."

    context_text = "\n\n---\n\n".join(
        chunks_df["CHUNK_TEXT"].tolist()
    )

    prompt = f"""
You are a medical document assistant.

Provide complete details about the {entity_type} "{name}".
Use ONLY the context below.
Do not hallucinate.

Context:
{context_text}

Answer:
"""

    return call_llm(MODEL_NAME, prompt)

# -----------------------------------------------------------------------------
# LOGIN SCREEN
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
# ADMIN / OWNER DATA EXPLORER
# -----------------------------------------------------------------------------
if st.session_state.app_role in ["admin", "owner"]:

    st.sidebar.markdown("## 🔍 Data Explorer")

    category_map = {
        "Patient Details": "patient",
        "Doctor Details": "doctor",
        "Hospital Details": "hospital"
    }

    selected_category = st.sidebar.radio(
        "Select Category",
        list(category_map.keys())
    )

    entity_type = category_map[selected_category]

    entities = extract_entities(entity_type)

    if not entities:
        st.sidebar.info("No validated entities found.")
    else:
        selected_entity = st.sidebar.selectbox(
            f"Select {entity_type.title()}",
            ["-- Select --"] + entities
        )

        if selected_entity != "-- Select --":

            answer = generate_entity_answer(
                selected_entity,
                entity_type
            )

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

# -----------------------------------------------------------------------------
# MAIN CHAT WINDOW
# -----------------------------------------------------------------------------
st.title("📄 PDF Chatbot on Snowflake")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_prompt = st.chat_input("Ask a question about the PDFs")

if user_prompt:

    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            chunks_df = vector_search(user_prompt, top_k=10)

            if chunks_df.empty:
                answer = "No relevant content found in documents."
            else:
                context_text = "\n\n---\n\n".join(
                    chunks_df["CHUNK_TEXT"].tolist()
                )

                full_prompt = f"""
Answer the question using ONLY the context below.
Do not hallucinate.

Context:
{context_text}

Question:
{user_prompt}

Answer:
"""

                answer = call_llm(MODEL_NAME, full_prompt)

            st.write(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
