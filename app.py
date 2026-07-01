from modules.pdf_processor import extract_text_from_pdf
from modules.text_chunker import create_chunks
from modules.summarizer import hierarchial_summarizer
from modules.embedding_generator import generate_embedding,generate_embeddings
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
    chunks = create_chunks(extracted_text)
    """
    final_summary=hierarchial_summarizer(createdChunks)
    st.subheader("Final Summary")
    st.write(final_summary)
    """
    embedding = generate_embedding(chunks[0])

    st.subheader("Embedding Information")

    st.write("Number of chunks: ",len(chunks))
    st.write("Embedding Type: ",type(embedding))
    st.write("Embedding Shape: ",embedding.shape)

    st.subheader("First Chunk")
    st.write(chunks[0])

    st.subheader("Fiest 20 Embedding Values")
    st.write(embedding[:20])

