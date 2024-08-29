import streamlit as st

from utility.handler import get_tokenizer_and_model

get_tokenizer_and_model()

st.title("RAG Validator")
st.write(
    "This application can validate the RAG related technologies like PDF Loader, NLP Embedding ... etc."
)
if st.button(label="PDF Loader"):
    st.switch_page("pages/pdf_loader.py")
if st.button(label="Embedding Checker"):
    st.switch_page("pages/embedding_checker.py")
if st.button(label="PDF Embedding"):
    st.switch_page("pages/pdf_embedding.py")
if st.button(label="Compare PDF"):
    st.switch_page("pages/compare_pdf.py")
