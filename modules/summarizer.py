import streamlit as st
from transformers import pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization",model = "t5-small")
summarizer = load_summarizer()
def summarize_chunk(chunk):
    result = summarizer(chunk,max_length=150,min_length=40,do_sample=False)
    return result[0]["summary_text"]
def summarize_document(chunks):
    chunk_summaries=[]
    for chunk in chunks:
        summary = summarize_chunk(chunk)
        chunk_summaries.append(summary)
    combined_summary = "\n".join(chunk_summaries)
    final_summary = summarize_chunk(combined_summary)
    return final_summary