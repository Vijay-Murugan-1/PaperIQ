from modules.pdf_processor import extract_text_from_pdf
from modules.text_chunker import create_chunks
from modules.summarizer import summarize_document
import streamlit as st
import pymupdf
def main():
    st.title("PaperIQ!")
    st.subheader("AI-based research assistant for papers.")
    st.write("PaperIQ is an AI-powered research assistant designed to help you quickly understand and analyze academic papers. Simply upload a paper, and PaperIQ will provide you with a concise summary, key insights, and answers to your questions about the content. Whether you're a student, researcher, or just curious, PaperIQ makes it easier than ever to stay informed and up-to-date with the latest research.")
if __name__ == "__main__":    main()    
uploaded_file = st.file_uploader("Upload a paper",type=["pdf"])
if uploaded_file is not None:
    st.write(f"{uploaded_file.name} uploaded successfully!")
    pdf_bytes = uploaded_file.getvalue()
    extracted_text = extract_text_from_pdf(pdf_bytes)
    createdChunks = create_chunks(extracted_text)
    final_summary=summarize_document(createdChunks)
    st.subheader("Final Summary")
    st.write(final_summary)

