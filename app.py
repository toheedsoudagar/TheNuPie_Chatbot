# app.py
import streamlit as st
import pandas as pd
from rag import RAGPipeline 

# ---------- CONFIGURATION ----------
PAGE_TITLE = "NuPIE Insights"
HEADER_TITLE = "NuPIE Insights: The Offline Chatbot"
LOGO_PATH = "nu_pie.svg" 
# -----------------------------------

st.set_page_config(page_title=PAGE_TITLE, layout="centered", page_icon=LOGO_PATH)

# --- CUSTOM HEADER ---
# FIX: Use vertical_alignment="center" to align logo and text perfectly
col1, col2 = st.columns([1, 6], gap="small", vertical_alignment="center") 

with col1:
    try:
        st.image(LOGO_PATH, width=90)
    except Exception:
        st.error("Logo missing")

with col2:
    # FIX: Use markdown with custom styling to remove the huge default top-margin of st.title
    st.markdown(
        f"<h1 style='margin-top: 0; padding-top: 0;'>{HEADER_TITLE}</h1>", 
        unsafe_allow_html=True
    )

st.divider()

# --- Pipeline Initialization ---
@st.cache_resource
def initialize_pipeline():
    return RAGPipeline()

pipeline = initialize_pipeline()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    

# --- Message Rendering ---
def render_message(role, content, sources=None):
    if role == "user":
        avatar = "👤" 
    else:
        avatar = LOGO_PATH 

    with st.chat_message(role, avatar=avatar):
        st.markdown(content)
        
        if sources:
            with st.expander("View Sources / Raw Data"):
                for i, s in enumerate(sources):
                    st.caption(f"**Source {i+1}:** `{s.get('source', 'Unknown')}`")
                    
                    if s.get("type") == "sql":
                        try:
                            df = pd.DataFrame(s["content"])
                            st.dataframe(df)
                        except Exception:
                            st.code(str(s["content"]))
                    else:
                        st.text(s["content"])
                    
                    st.divider()

# Loop through history
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("sources"))

# --- Chat Input ---
query = st.chat_input("Ask NuPIE Insights...")

if query:
    st.session_state.messages.append({"role": "user", "content": query, "sources": []})
    render_message("user", query)

    with st.spinner("Analyzing databases and documents..."):
        answer, sources = pipeline.ask(query)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    render_message("assistant", answer, sources)