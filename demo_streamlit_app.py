import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

session = get_active_session()
STAGE_NAME = "AI_POC_DB.PII_PHI_POC.PHI_PII_POC_STAGE1"

TOP_K = 10
MODEL_NAME = "llama3.1-70b"
SIMILARITY_THRESHOLD = 0.65

# -----------------------------------------------------------------------------
# SESSION STATE INIT
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

if "entity_cache" not in st.session_state:
    st.session_state.entity_cache = {}

# -----------------------------------------------------------------------------
# AUTHENTICATION
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
# LLM CALL
# -----------------------------------------------------------------------------
def call_llm(model, prompt):
    sql = "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER"
    row = session.sql(sql, params=[model, prompt]).collect()[0]
    return row["ANSWER"]

# -----------------------------------------------------------------------------
# VECTOR SEARCH
# -----------------------------------------------------------------------------
def call_search(query, k):

    sql = f"""
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

    return session.sql(sql, params=[query]).to_pandas()

# -----------------------------------------------------------------------------
# MASK ANSWER (FOR NON-ADMIN)
# -----------------------------------------------------------------------------
def mask_answer(answer):

    masking_prompt = f"""
You are a healthcare data privacy engine.

Mask ALL PII and PHI.

Rules:
- Replace sensitive values with exactly "XXXXXX"
- Do not explain
- Return only masked text

Text:
{answer}

Masked Output:
"""

    return call_llm(MODEL_NAME, masking_prompt)

# -----------------------------------------------------------------------------
# PRESIGNED URL
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
# ENTITY EXTRACTION USING LLM
# -----------------------------------------------------------------------------
def extract_entities(entity_type):

    if entity_type in st.session_state.entity_cache:
        return st.session_state.entity_cache[entity_type]

    df = session.sql("""
        SELECT CHUNK_TEXT
        FROM DOCS_CHUNKS
        LIMIT 1000
    """).to_pandas()

    combined_text = "\n".join(df["CHUNK_TEXT"].tolist())

    prompt = f"""
Extract all unique {entity_type} names from the medical text below.

Rules:
- Return ONLY comma separated names
- No duplicates
- No explanation
- Clean format

Text:
{combined_text}

Output:
"""

    response = call_llm(MODEL_NAME, prompt)

    names = [n.strip() for n in response.split(",") if n.strip()]
    names = sorted(list(set(names)))

    st.session_state.entity_cache[entity_type] = names

    return names

# -----------------------------------------------------------------------------
# GENERATE CHAT ANSWER
# -----------------------------------------------------------------------------
def generate_answer(question):

    chunks_df = call_search(question, TOP_K)

    if chunks_df.empty:
        return "No relevant content found."

    context = "\n\n---\n\n".join(chunks_df["CHUNK_TEXT"].tolist())

    prompt = f"""
You are a medical document assistant.

Answer ONLY using context below.
Be precise.
Do not hallucinate.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = call_llm(MODEL_NAME, prompt)

    # Mask for non-admin
    if st.session_state.app_role not in ["admin", "owner"]:
        answer = mask_answer(answer)

    return answer, chunks_df

# -----------------------------------------------------------------------------
# LOGIN SCREEN
# -----------------------------------------------------------------------------
if not st.session_state.authenticated:

    st.title("🔐 Chatbot Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

    if submit:

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
st.sidebar.write("👤", st.session_state.username)
st.sidebar.write("🛡️", st.session_state.app_role.upper())

if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

# -----------------------------------------------------------------------------
# ADMIN / OWNER ENTITY TABS
# -----------------------------------------------------------------------------
if st.session_state.app_role in ["admin", "owner"]:

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Entity Explorer")

    category = st.sidebar.radio(
        "Select Category",
        ["Patient Details", "Doctor Details", "Hospital Details"]
    )

    entity_map = {
        "Patient Details": "patient",
        "Doctor Details": "doctor",
        "Hospital Details": "hospital"
    }

    selected_type = entity_map[category]

    search_text = st.sidebar.text_input(f"Search {selected_type}")

    names = extract_entities(selected_type)

    if search_text:
        names = [n for n in names if search_text.lower() in n.lower()]

    for name in names[:50]:

        if st.sidebar.button(name, key=f"{selected_type}_{name}"):

            question = f"Provide complete details about {name}."
            st.session_state.messages.append({"role": "user", "content": question})

            answer, chunks_df = generate_answer(question)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            st.rerun()

# -----------------------------------------------------------------------------
# MAIN CHAT
# -----------------------------------------------------------------------------
st.title("📄 PDF Chatbot on Snowflake")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask a question about your PDFs")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})

    answer, chunks_df = generate_answer(prompt)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Admin: Show best matching PDF
    if st.session_state.app_role in ["admin", "owner"]:

        file_scores = (
            chunks_df
            .groupby("SOURCE_FILE")["SCORE"]
            .mean()
            .reset_index()
            .sort_values("SCORE", ascending=False)
        )

        if not file_scores.empty:

            best_file = file_scores.iloc[0]["SOURCE_FILE"]
            best_score = file_scores.iloc[0]["SCORE"]

            if best_score >= SIMILARITY_THRESHOLD:

                with st.chat_message("assistant"):
                    st.markdown("### 📥 Most Relevant PDF")
                    url = get_presigned_url(best_file)
                    st.link_button(
                        f"Download {best_file} (Score: {best_score:.3f})",
                        url
                    )

    st.rerun()
