import streamlit as st

from modules.context_builder import build_context
from modules.vector_store import (
    create_index,
    add_embeddings,
    search_index
)

from modules.embedding_generator import generate_embedding,generate_embeddings
from modules.llm import generate_response
from modules.pdf_processor import extract_text_from_pdf
from modules.prompt_builder import build_prompt
from modules.insight_builder import build_insight_prompt
from modules.summary_builder import build_summary_prompt
from modules.text_chunker import create_chunks

if "paper" not in st.session_state:
    st.session_state.paper = None
def main() -> None:
    st.title("PaperIQ!")
    st.subheader("AI-based research assistant for papers.")
    st.write("PaperIQ is an AI-powered research assistant designed to help you quickly understand and analyze academic papers. Simply upload a paper, and PaperIQ will provide you with a concise summary, key insights, and answers to your questions about the content. Whether you're a student, researcher, or just curious, PaperIQ makes it easier than ever to stay informed and up-to-date with the latest research.") 
    uploaded_file = st.file_uploader("Upload a paper",type=["pdf"])
    if uploaded_file is not None:
        st.write(f"{uploaded_file.name} uploaded successfully.")
        pdf_bytes = uploaded_file.getvalue()
        paper = st.session_state.paper
        needs_processing = (paper is None or paper["name"] != uploaded_file.name)
        if needs_processing:
            with st.spinner("Processing Paper..."):

                extracted_text = extract_text_from_pdf(pdf_bytes)
                if not extracted_text.strip():
                    st.error("No text could be extracted from the uploaded PDF.")
                    st.stop()
                    
                chunks = create_chunks(extracted_text)
                if not chunks:
                    st.error("Unable to create text chunks from the document.")
                    st.stop()

                embeddings = generate_embeddings(chunks)
                dimension = embeddings[0].shape[0]
                index = create_index(dimension)
                add_embeddings(index,embeddings)

                st.session_state.paper = {
                    "name": uploaded_file.name,
                    "text": extracted_text,
                    "chunks": chunks,
                    "embeddings": embeddings,
                    "index": index,
                }

            st.success("Paper processed successfully!")
            st.divider()

        paper = st.session_state.paper
        extracted_text = paper["text"]
        chunks = paper["chunks"]
        embeddings = paper["embeddings"]
        index = paper["index"]

        feature = st.radio(
            "Choose a feature",
            (
                "Summary",
                "Ask Questions",
                "Key Insights",
                "Quiz Generator",
                "Explain Concepts",
                "Flashcards"
            )
        )
        if feature == "Summary":
            if st.button("Generate Summary"):
                summary_prompt = build_summary_prompt(extracted_text)

                with st.spinner("Generating summary..."):
                    
                    try:

                        summary = generate_response(summary_prompt)
                        st.subheader("Paper Summary")
                        st.write(summary)

                    except Exception as error:

                        st.error(f"Failed to generate summary: {error}")
            
        elif feature == "Ask Questions":
            

            user_query = st.text_input("Ask a question about the paper")

            if user_query:

                query_embedding = generate_embedding(user_query)
                distances, indices = search_index(index,query_embedding,top_k=3)

                context = build_context(chunks,indices)
                
                prompt = build_prompt(context = context,
                                    question = user_query)
                
                with st.spinner("Generating answer..."):
                    try:

                        answer = generate_response(prompt)
                        st.subheader("Answer")
                        st.write(answer)

                    except Exception as error:

                        st.error(f"Failed to answer question: {error}")

        elif feature == "Key Insights":

            if st.button("Generate Insights"):

                insight_prompt = build_insight_prompt(extracted_text)
                try:
                    with st.spinner("Generating key insights..."):
                        insights = generate_response(insight_prompt)
                    
                    st.subheader("Key Insights")
                    st.write(insights)
                    
                except Exception as error:
                    st.error(f"Failed to generate insights: {error}")

        elif feature == "Quiz Generator":

            if st.button("Generate Quiz"):

                st.info("Coming soon.")

        elif feature == "Explain Concepts":

            st.info("Coming soon.")

        elif feature == "Flashcards":
            if st.button("Generate Flashcards"):
                st.info("Coming soon.")

if __name__ == "__main__":    
    main()   