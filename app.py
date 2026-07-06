from modules.pdf_processor import extract_text_from_pdf
from modules.text_chunker import create_chunks
from modules.embedding_generator import generate_embedding,generate_embeddings
import streamlit as st
from modules.vector_store import (
    create_index,
    add_embeddings,
    search_index
)
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
    embeddings = generate_embeddings(chunks)
    dimension = embeddings[0].shape[0]
    index = create_index(dimension)
    add_embeddings(index,embeddings)
    st.success("Vector database created successfully!")

    st.write("Number of Chunks:", len(chunks))
    st.write("Number of Embeddings:", len(embeddings))
    st.write("Embedding Dimension:", dimension)
    st.write("Vectors in Index:", index.ntotal)
    
    st.divider()
    user_query = st.text_input("Ask a question about the paper")

    if user_query:
        query_embedding = generate_embedding(user_query)
        distances, indices = search_index(index,query_embedding,top_k=5)

        st.subheader("Retrieved Chunks")

        for rank,chunk_index in enumerate(indices,start=1):
            st.markdown(f"### Result {rank}")
            st.write(chunks[chunk_index])

            st.caption(f"Chunk Index: {chunk_index}")
            st.caption(f"L2 Distance: {distances[rank-1]:.4f}")
    st.divider()
