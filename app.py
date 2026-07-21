import streamlit as st
import json

from modules.context_builder import build_context
from modules.vector_store import (
    create_index,
    add_embeddings,
    search_index
)

from modules.embedding_generator import generate_embedding, generate_embeddings
from modules.llm import generate_response
from modules.pdf_processor import extract_text_from_pdf
from modules.prompt_builder import build_prompt
from modules.insight_builder import build_insight_prompt
from modules.summary_builder import build_summary_prompt
from modules.flashcard_builder import build_flashcard
from modules.quiz_builder import build_quiz_prompt
from modules.text_chunker import create_chunks

if "paper" not in st.session_state:
    st.session_state.paper = None

def main() -> None:
    st.title("PaperIQ!")
    st.subheader("AI-based research assistant for papers.")
    st.write(
        "PaperIQ is an AI-powered research assistant designed to help you quickly understand and analyze "
        "academic papers. Simply upload a paper, and PaperIQ will provide you with a concise summary, key insights, "
        "and answers to your questions about the content. Whether you're a student, researcher, or just curious, "
        "PaperIQ makes it easier than ever to stay informed and up-to-date with the latest research."
    ) 
    
    uploaded_file = st.file_uploader("Upload a paper", type=["pdf"])
    
    if uploaded_file is not None:
        st.write(f"{uploaded_file.name} uploaded successfully.")
        pdf_bytes = uploaded_file.getvalue()
        paper = st.session_state.paper
        needs_processing = (paper is None or paper["name"] != uploaded_file.name)
        
        if needs_processing:
            with st.spinner("Processing Paper..."):

                # 1. Extract page data (list of dicts containing 'page' and 'text')
                pages_data = extract_text_from_pdf(pdf_bytes)
                if not pages_data:
                    st.error("No text could be extracted from the uploaded PDF.")
                    st.stop()
                
                # Full string representation for whole-paper tasks (Summary, Insights, etc.)
                full_text = "\n\n".join([page["text"] for page in pages_data])

                # 2. Token-based chunking with page metadata
                chunks = create_chunks(pages_data)
                if not chunks:
                    st.error("Unable to create text chunks from the document.")
                    st.stop()

                # 3. Generate embeddings using only the text field of each chunk
                chunk_texts = [chunk["text"] for chunk in chunks]
                embeddings = generate_embeddings(chunk_texts)
                
                dimension = embeddings[0].shape[0]
                index = create_index(dimension)
                add_embeddings(index, embeddings)

                # Store in session state
                st.session_state.paper = {
                    "name": uploaded_file.name,
                    "text": full_text,
                    "pages_data": pages_data,
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
                with st.spinner("Generating answer..."):
                    try:
                        # 1. Search FAISS vector store
                        query_embedding = generate_embedding(user_query)
                        distances, indices = search_index(index, query_embedding, top_k=3)

                        # Flatten FAISS 2D index array if needed
                        retrieved_indices = indices[0] if indices.ndim > 1 else indices

                        # 2. Build context with page numbers
                        context = build_context(chunks, retrieved_indices)
                        
                        prompt = build_prompt(
                            context=context,
                            question=user_query
                        )
                        
                        # 3. Generate answer
                        answer = generate_response(prompt)
                        st.subheader("Answer")
                        st.write(answer)

                        # 4. Source Citations UI
                        st.write("")
                        with st.expander("🔍 View Sources"):
                            matched_chunks = [chunks[idx] for idx in retrieved_indices if idx < len(chunks)]
                            source_pages = sorted(list(set(chunk["page"] for chunk in matched_chunks)))

                            st.write("**Information retrieved from:**")
                            if source_pages:
                                cols = st.columns(len(source_pages))
                                for i, page_num in enumerate(source_pages):
                                    with cols[i]:
                                        st.info(f"📄 Page {page_num}")
                            else:
                                st.write("No specific page metadata found.")

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
            num_q = st.slider("Number of questions", min_value=3, max_value=10, value=5)

            if st.button("Generate Quiz"):
                quiz_prompt = build_quiz_prompt(extracted_text, num_q)
                
                with st.spinner("Generating quiz..."):
                    try:
                        raw_response = generate_response(quiz_prompt)
                        cleaned_response = raw_response.replace("```json", "").replace("```", "").strip()
                        
                        st.session_state.quiz_data = json.loads(cleaned_response)
                        st.session_state.show_results = False
                        st.session_state.user_answers = {}
                        
                    except json.JSONDecodeError:
                        st.error("Failed to parse the quiz format. Please try generating again.")
                    except Exception as error:
                        st.error(f"Failed to generate quiz: {error}")

            if "quiz_data" in st.session_state:
                st.subheader("Test Your Knowledge")
                
                with st.form("quiz_form"):
                    user_answers = {}
                    for i, q in enumerate(st.session_state.quiz_data):
                        st.write(f"**Q{i+1}: {q['question']}**")

                        user_answers[i] = st.radio(
                            "Options", 
                            q["options"], 
                            key=f"q_{i}", 
                            label_visibility="collapsed", 
                            index=None
                        )
                        st.write("---")
                    
                    submitted = st.form_submit_button("Submit Answers")
                    
                    if submitted:
                        st.session_state.user_answers = user_answers
                        st.session_state.show_results = True

                if st.session_state.get("show_results"):
                    score = 0
                    for i, q in enumerate(st.session_state.quiz_data):
                        user_ans = st.session_state.user_answers.get(i)
                        
                        if user_ans == q["correct_answer"]:
                            score += 1
                            st.success(f"**Q{i+1} Correct!**\n\n{q['explanation']}")
                        else:
                            st.error(f"**Q{i+1} Incorrect.**\n\n**You chose:** {user_ans}\n\n**Correct answer:** {q['correct_answer']}\n\n*Explanation: {q['explanation']}*")
                    
                    st.write(f"### Final Score: {score} / {len(st.session_state.quiz_data)}")
                    
                    if st.button("Clear Quiz"):
                        del st.session_state.quiz_data
                        st.rerun()

        elif feature == "Flashcards":
            num_cards = st.slider("Number of Flashcards", min_value=5, max_value=20, value=10)
            
            if st.button("Generate Flashcards"):
                flashcard_prompt = build_flashcard(extracted_text, num_cards)

                with st.spinner("Extracting key concepts for flashcards..."):
                    try:
                        raw_response = generate_response(flashcard_prompt)
                        cleaned_response = raw_response.replace("```json", "").replace("```", "").strip()

                        st.session_state.flashcards_data = json.loads(cleaned_response)
                        st.session_state.current_card = 0
                        st.session_state.is_flipped = False
                    except json.JSONDecodeError:
                        st.error("Failed to parse the flashcard format. Please try generating again.")
                    except Exception as error:
                        st.error(f"Failed to generate flashcards: {error}")
            
            if "flashcards_data" in st.session_state:
                cards = st.session_state.flashcards_data
                idx = st.session_state.current_card
                
                st.markdown(f"### Card {idx + 1} of {len(cards)}")
                
                current_card_data = cards[idx]
                
                if not st.session_state.is_flipped:
                    st.info(f"**Front (Concept / Question):**\n\n### {current_card_data['front']}")
                else:
                    st.success(f"**Back (Definition / Answer):**\n\n{current_card_data['back']}")

                st.write("") # Spacer

                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("⬅️ Previous", use_container_width=True, disabled=(idx == 0)):
                        st.session_state.current_card -= 1
                        st.session_state.is_flipped = False
                        st.rerun()
                        
                with col2:
                    if st.button("🔄 Flip Card", use_container_width=True):
                        st.session_state.is_flipped = not st.session_state.is_flipped
                        st.rerun()
                        
                with col3:
                    if st.button("Next ➡️", use_container_width=True, disabled=(idx == len(cards) - 1)):
                        st.session_state.current_card += 1
                        st.session_state.is_flipped = False
                        st.rerun()

                st.divider()
                
                if st.button("Clear Deck"):
                    del st.session_state.flashcards_data
                    st.rerun()

if __name__ == "__main__":    
    main()