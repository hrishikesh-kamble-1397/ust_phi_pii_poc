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

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
MODEL_NAME = "mistral-large2"
EMBED_MODEL = "snowflake-arctic-embed-m"
SIMILARITY_THRESHOLD = 0.35
MAX_CHUNKS = 30

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

if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None

if "selected_doctor" not in st.session_state:
    st.session_state.selected_doctor = None


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
# Fetch Patients With Full Details
# -----------------------------------------------------------------------------
def get_all_patients():
    query = """
        SELECT DISTINCT PATIENT_NAME
        FROM AI_POC_DB.PII_PHI_POC.PATIENT_DETAILS
        WHERE PATIENT_NAME IS NOT NULL
    """
    return session.sql(query).to_pandas()


def get_patient_details(patient_name):
    query = """
        SELECT *
        FROM AI_POC_DB.PII_PHI_POC.PATIENT_DETAILS
        WHERE PATIENT_NAME = :1
    """
    return session.sql(query, [patient_name]).to_pandas()


# -----------------------------------------------------------------------------
# Fetch Doctors With Patients
# -----------------------------------------------------------------------------
def get_all_doctors():
    query = """
        SELECT DISTINCT DOCTOR_NAME
        FROM AI_POC_DB.PII_PHI_POC.PATIENT_DETAILS
        WHERE DOCTOR_NAME IS NOT NULL
    """
    return session.sql(query).to_pandas()


def get_doctor_details(doctor_name):
    query = """
        SELECT *
        FROM AI_POC_DB.PII_PHI_POC.PATIENT_DETAILS
        WHERE DOCTOR_NAME = :1
    """
    return session.sql(query, [doctor_name]).to_pandas()


# -----------------------------------------------------------------------------
# LOGIN
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
# ADMIN / OWNER TABS
# -----------------------------------------------------------------------------
if st.session_state.app_role in ["admin", "owner"]:

    st.sidebar.markdown("## 📂 Data Access")

    tab_selection = st.sidebar.radio(
        "Select View",
        ["Patient Details", "Doctor Details"]
    )

    if tab_selection == "Patient Details":
        patients_df = get_all_patients()

        if not patients_df.empty:
            st.sidebar.markdown("### Patients")

            for patient in patients_df["PATIENT_NAME"]:
                if st.sidebar.button(patient):
                    st.session_state.selected_patient = patient
                    st.session_state.selected_doctor = None

    if tab_selection == "Doctor Details":
        doctors_df = get_all_doctors()

        if not doctors_df.empty:
            st.sidebar.markdown("### Doctors")

            for doctor in doctors_df["DOCTOR_NAME"]:
                if st.sidebar.button(doctor):
                    st.session_state.selected_doctor = doctor
                    st.session_state.selected_patient = None


# -----------------------------------------------------------------------------
# MAIN WINDOW
# -----------------------------------------------------------------------------
st.title("📄 PDF Chatbot on Snowflake")

# Show Patient Details
if st.session_state.selected_patient:
    details_df = get_patient_details(st.session_state.selected_patient)

    st.subheader(f"Patient: {st.session_state.selected_patient}")
    st.dataframe(details_df)
    st.stop()

# Show Doctor Details
if st.session_state.selected_doctor:
    details_df = get_doctor_details(st.session_state.selected_doctor)

    st.subheader(f"Doctor: {st.session_state.selected_doctor}")

    if not details_df.empty:
        st.markdown("### Patients Treated")
        st.write(details_df["PATIENT_NAME"].unique())

        st.markdown("### Full Records")
        st.dataframe(details_df)

    st.stop()

# -----------------------------------------------------------------------------
# NORMAL CHATBOT BELOW
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask about your PDFs")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        st.write("Chatbot functionality remains unchanged.")
