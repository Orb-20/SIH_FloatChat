import streamlit as st
import requests

st.set_page_config(page_title="FloatChat-ARGO", layout="wide")

st.title("🌊 FloatChat-ARGO (Scaffold)")
st.write("This is a minimal UI scaffold. We'll wire data and maps in the next steps.")

with st.sidebar:
    st.header("Ask the ocean 🤖")
    question = st.text_input("Your question", placeholder="Show me salinity profiles near the equator in March 2023")
    if st.button("Ask"):
        try:
            resp = requests.post("http://localhost:8001/ask", json={"question": question}, timeout=10)
            st.success(resp.json())
        except Exception as e:
            st.error(f"API not running yet: {e}")
