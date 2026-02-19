import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

# -----------------------------------------------------------------------------
# APP CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PDF + Database Chatbot", page_icon="📄", layout="wide")

session = get_active_session()

SIMILARITY_THRESHOLD = 0.70
MODEL_NAME = "llama3.1-70b"

# -----------------------------------------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your PDFs or Database."}
    ]

# -----------------------------------------------------------------------------
# SIDEBAR TOGGLE
# -----------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Data Source")

data_mode = st.sidebar.toggle(
    "Use Database Instead of PDFs",
    value=False
)

if data_mode:
    st.sidebar.success("🗄️ Database Mode Enabled")
else:
    st.sidebar.success("📄 PDF Mode Enabled")

# -----------------------------------------------------------------------------
# HELPER: GET CURRENT ROLE (RBAC)
# -----------------------------------------------------------------------------
def get_current_role():
    role_df = session.sql("SELECT CURRENT_ROLE()").collect()
    return role_df[0][0]

# -----------------------------------------------------------------------------
# LLM CALL (CORTEX COMPLETE)
# -----------------------------------------------------------------------------
def call_llm(prompt):
    sql = """SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS ANSWER"""
    row = session.sql(sql, params=[MODEL_NAME, prompt]).collect()[0]
    return row["ANSWER"]

# -----------------------------------------------------------------------------
# PDF VECTOR SEARCH
# -----------------------------------------------------------------------------
def search_pdf_context(query):

    sql = """
        WITH query_vec AS (
            SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768(
                'snowflake-arctic-embed-m',
                ?
            ) AS emb
        )
        SELECT
            CHUNK_TEXT,
            VECTOR_COSINE_SIMILARITY(EMBEDDING, q.emb) AS SCORE
        FROM DOCS_CHUNKS, query_vec q
        ORDER BY SCORE DESC
        LIMIT 5
    """

    df = session.sql(sql, params=[query]).to_pandas()

    if df.empty:
        return None, 0

    best_score = df.iloc[0]["SCORE"]

    if best_score < SIMILARITY_THRESHOLD:
        return None, best_score

    context = "\n\n---\n\n".join(df["CHUNK_TEXT"].tolist())

    return context, best_score

# -----------------------------------------------------------------------------
# DATABASE MODE (TEXT OUTPUT WITH RBAC)
# -----------------------------------------------------------------------------
def search_database_context(user_query):

    # Step 1: Convert Question → SQL
    sql_prompt = f"""
    Convert this question into SQL query.
    Use only PATIENT_DATA table.
    Do not explain.
    Return only SQL.

    Question:
    {user_query}

    SQL:
    """

    sql_query = call_llm(sql_prompt)

    try:
        # Step 2: Execute SQL (Snowflake handles RBAC + Masking)
        df = session.sql(sql_query).to_pandas()

        if df.empty:
            return None

        # Step 3: Convert records into JSON text context
        data_context = df.to_json(orient="records")

        # Step 4: Convert structured data → natural language
        answer_prompt = f"""
        You are a medical assistant.

        Using ONLY the database records below,
        answer clearly in paragraph format.

        Do not hallucinate.
        Do not mention SQL.
        Do not show table format.

        Database Records:
        {data_context}

        Question:
        {user_query}

        Answer:
        """

        final_answer = call_llm(answer_prompt)

        return final_answer

    except Exception as e:
        return f"Database Error: {str(e)}"

# -----------------------------------------------------------------------------
# MAIN UI
# -----------------------------------------------------------------------------
st.title("📄 PDF + Database Chatbot")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("Ask your question")

if user_input:

    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if data_mode:
                # DATABASE MODE
                answer = search_database_context(user_input)

                if not answer:
                    answer = "No relevant database information found."

            else:
                # PDF MODE
                context, score = search_pdf_context(user_input)

                if not context:
                    answer = "No relevant content found in PDFs."
                else:
                    prompt = f"""
                    Answer using ONLY the context below.
                    Do not hallucinate.

                    Context:
                    {context}

                    Question:
                    {user_input}

                    Answer:
                    """
                    answer = call_llm(prompt)

            st.write(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

# -----------------------------------------------------------------------------
# FOOTER (OPTIONAL DEBUG INFO)
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔐 Security Info")
st.sidebar.write("Current Role:", get_current_role())
