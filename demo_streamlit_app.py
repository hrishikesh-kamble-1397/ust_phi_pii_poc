import re
import json
import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session

# -----------------------------------------------------------------------------
# App config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")
st.title("📄 PDF Chatbot on Snowflake")

# Obtain active Snowflake session (works in Snowsight / Snowflake-hosted Streamlit)
session = get_active_session()

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("Settings")

# LLM model selector (Snowflake Cortex COMPLETE)
model = st.sidebar.selectbox(
    "LLM model",
    options=[
        "mistral-large",
        "llama3.1-70b",
        "mixtral-8x7b"
    ],
    index=0
)

top_k = st.sidebar.slider("Top‑K chunks", min_value=1, max_value=10, value=5, step=1)
show_sources = st.sidebar.checkbox("Show sources", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("PII/PHI Extractor (Optional)")
default_stage_file = "sample_hospitalization_claim1.pdf"
stage_file_name = st.sidebar.text_input(
    "Stage file name (file must be uploaded in the stage)",
    value=default_stage_file,
    help="The file must already be uploaded to the specified stage."
)
run_proc = st.sidebar.button("Run PII/PHI Parse & Classify")

# -----------------------------------------------------------------------------
# Helpers: DB calls
# -----------------------------------------------------------------------------
def call_search(query: str, k: int, service: str) -> pd.DataFrame:
    """
    Calls Snowflake Cortex Search to fetch top-k chunks from the given service.
    The service should be built on a masked/secure view.
    """
    search_sql = f"""
        SELECT
          CHUNK_TEXT,
          SOURCE_FILE,
          SCORE
        FROM SNOWFLAKE.CORTEX.SEARCH(
          SERVICE => %s,
          QUERY   => %s,
          TOP_K   => {k}
        )
        ORDER BY SCORE DESC
    """
    return session.sql(search_sql, params=[service, query]).to_pandas()

def call_llm(model_name: str, prompt: str) -> str:
    """
    Calls Snowflake Cortex COMPLETE to generate the answer.
    """
    llm_sql = "SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS ANSWER"
    row = session.sql(llm_sql, params=[model_name, prompt]).collect()[0]
    return row["ANSWER"]

def sanitize_suffix(file_name: str) -> str:
    """
    Mirrors the stored procedure's suffix normalization:
    - remove .pdf
    - replace non-alphanumeric with underscores
    - uppercase
    """
    base = re.sub(r"\.pdf$", "", file_name, flags=re.IGNORECASE)
    base = re.sub(r"[^A-Za-z0-9]+", "_", base)
    return base.upper()

def call_pii_phi_proc(file_name: str) -> str:
    """
    Calls your stored procedure and returns the procedure's message.
    """
    sql = "CALL AI_POC_DB.PII_PHI_POC.SP_PARSE_EXTRACT_CLASSIFY(%s)"
    row = session.sql(sql, params=[file_name]).collect()[0]
    return list(row.asDict().values())[0]

def load_output_table(file_name: str) -> pd.DataFrame:
    """
    Loads the output table produced by the procedure, reconstructing the FQN
    from the same suffix logic.
    """
    suffix = sanitize_suffix(file_name)
    out_fqn = f"AI_POC_DB.PII_PHI_POC.OUTPUT_{suffix}"
    return session.table(out_fqn).to_pandas()

# -----------------------------------------------------------------------------
# Helpers: RBAC/Redaction
# -----------------------------------------------------------------------------
def get_caller_entitlements():
    """
    Returns (caller_user, roles_set, can_view_pii, can_view_phi)
    Uses CURRENT_AVAILABLE_ROLES() to evaluate the caller's roles
    even when this app runs with owner's rights.
    """
    sql = "SELECT CURRENT_USER() AS U, CURRENT_AVAILABLE_ROLES() AS R"
    row = session.sql(sql).collect()[0]
    roles = set(json.loads(row["R"])) if row["R"] else set()
    can_view_phi = "PHI_READER" in roles
    can_view_pii = can_view_phi or ("PII_READER" in roles)
    return row["U"], roles, can_view_pii, can_view_phi

# Simple regex-based fallback redaction for PII patterns
PII_PATTERNS = [
    (r'(?i)\b\d{3}[- ]?\d{2}[- ]?\d{4}\b', '***-**-****'),   # SSN-like
    (r'(?i)\b[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}\b', '***@***'), # email
    (r'(?i)\b(\+?\d[\d -]{7,}\d)\b', '***-REDACTED-PHONE***')# phone (very simple)
]

def redact_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    out = s
    for pat, repl in PII_PATTERNS:
        out = re.sub(pat, repl, out)
    return out

def redact_using_output_table(file_name: str, text: str, allow_phi: bool, allow_pii: bool) -> str:
    """
    Stronger redaction: uses detected PII/PHI values from OUTPUT_* table to
    replace exact matches. Falls back to regex if table is unavailable.
    """
    try:
        df = load_output_table(file_name)
        if allow_phi:
            return text  # no redaction required

        # Hide PHI values completely; for PII-only access, keep non-PHI after regex masking.
        sens_to_block = ["PHI"] if allow_pii else ["PHI", "PII", "HIGH"]
        if "SENSITIVITY" in df.columns and "VALUE" in df.columns:
            vals = df[df["SENSITIVITY"].isin(sens_to_block)]["VALUE"].dropna().unique().tolist()
        else:
            vals = []
        redacted = text
        for v in vals:
            if isinstance(v, str) and v.strip():
                redacted = redacted.replace(v, "[REDACTED]")
        if not allow_pii:
            redacted = redact_text(redacted)
        return redacted
    except Exception:
        return redact_text(text)

# -----------------------------------------------------------------------------
# Chat history
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me anything about your PDFs."}
    ]

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------------------------------------------------------
# Optional: run the PII/PHI procedure from the sidebar
# -----------------------------------------------------------------------------
if run_proc:
    with st.spinner("Running PII/PHI parse & classify stored procedure..."):
        try:
            proc_msg = call_pii_phi_proc(stage_file_name)
            st.sidebar.success(proc_msg)

            # Try to load and preview the output with RBAC-aware masking
            try:
                out_df = load_output_table(stage_file_name)
                caller_user, roles, can_view_pii, can_view_phi = get_caller_entitlements()

                preview_df = out_df.copy()
                if not can_view_phi and "SENSITIVITY" in preview_df.columns:
                    preview_df = preview_df[preview_df["SENSITIVITY"] != "PHI"]

                if not can_view_pii and "VALUE" in preview_df.columns:
                    preview_df["VALUE"] = "[REDACTED]"

                st.subheader("PII/PHI Output Table Preview")
                st.caption(f"Viewer: {caller_user} | Roles: {', '.join(sorted(roles)) or '(none)'}")
                st.dataframe(preview_df, use_container_width=True, height=320)
            except Exception as e:
                st.warning(
                    "Procedure succeeded, but could not load output table yet. "
                    f"If this persists, check naming and permissions.\n\nDetails: {e}"
                )
        except Exception as e:
            st.sidebar.error(f"Procedure failed: {e}")

# -----------------------------------------------------------------------------
# Chat input and response (RBAC-aware)
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Type your question about the PDFs"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Determine entitlements of the *caller*
                caller_user, caller_roles, can_view_pii, can_view_phi = get_caller_entitlements()

                # 1) Retrieve top-k relevant chunks (from secure service/view)
                chunks_df = call_search(prompt, top_k, search_service)

                if chunks_df.empty:
                    answer = (
                        "I couldn't find anything relevant in the PDFs. "
                        "Try rephrasing or broadening your query."
                    )
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    # Build context blocks with RBAC-aware redaction BEFORE calling the LLM
                    context_blocks = []
                    for _, row in chunks_df.iterrows():
                        src = row.get("SOURCE_FILE", "")
                        txt = row.get("CHUNK_TEXT", "")
                        score = row.get("SCORE", 0.0)

                        # Defense-in-depth redaction:
                        #   - If viewer lacks PHI, replace known PHI values from OUTPUT table
                        #   - If viewer lacks PII as well, regex mask residual hints
                        if not can_view_phi:
                            txt = redact_using_output_table(src, txt, allow_phi=can_view_phi, allow_pii=can_view_pii)

                        context_blocks.append(
                            f"File: {src}\nScore: {score:.4f}\nContent:\n{txt}\n"
                        )

                    context_text = "\n\n---\n\n".join(context_blocks)

                    system_prompt = (
                        "You are a helpful assistant that answers questions using only the provided PDF excerpts.\n"
                        "If the answer is not contained in the context, say you don’t know.\n"
                        "Cite the file names you used."
                    )

                    full_prompt = f"""{system_prompt}

Context:
{context_text}

Question: {prompt}
Answer:"""

                    # 2) Call LLM for the grounded answer
                    answer = call_llm(model, full_prompt)

                    # Output answer
                    st.caption(f"Viewer: {caller_user} | Roles: {', '.join(sorted(caller_roles)) or '(none)'}")
                    st.write(answer)

                    # Optional: show sources used (scoped by entitlements)
                    if show_sources:
                        if can_view_phi:
                            with st.expander("Show retrieved sources"):
                                st.dataframe(
                                    chunks_df[["SOURCE_FILE", "SCORE", "CHUNK_TEXT"]],
                                    use_container_width=True, height=300
                                )
                        elif can_view_pii:
                            with st.expander("Show retrieved sources (text masked)"):
                                masked = chunks_df.copy()
                                masked["CHUNK_TEXT"] = masked["CHUNK_TEXT"].apply(redact_text)
                                st.dataframe(
                                    masked[["SOURCE_FILE", "SCORE", "CHUNK_TEXT"]],
                                    use_container_width=True, height=300
                                )
                        else:
                            with st.expander("Show retrieved sources (no text)"):
                                st.dataframe(
                                    chunks_df[["SOURCE_FILE", "SCORE"]],
                                    use_container_width=True, height=300
                                )

                    st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                err_msg = f"Something went wrong while answering: {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
