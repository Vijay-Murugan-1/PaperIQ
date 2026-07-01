import streamlit as st
from transformers import pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization",model = "sshleifer/distilbart-cnn-12-6")
summarizer = load_summarizer()

def summarize_chunk(chunk):
    result = summarizer(chunk,max_length=120,min_length=50,do_sample=False)
    return result[0]["summary_text"]

def summarize_final(chunk):
    result = summarizer(chunk,max_length=300,min_length=150,do_sample=False)
    return result[0]["summary_text"]


def summarize_document(chunks):
    chunk_summaries=[]
    for chunk in chunks:
        summary = summarize_chunk(chunk)
        chunk_summaries.append(summary)
    combined_summary = "\n".join(chunk_summaries)
    final_summary = summarize_chunk(combined_summary)
    return final_summary

def hierarchial_summarizer(chunks,batch_size = 4):
    summaries = []
    for chunk in chunks:
        summaries.append(summarize_chunk(chunk))
    while len(summaries) > 1:
        next_level = []

        for i in range(0,len(summaries),batch_size):
            batch = summaries[i:i+batch_size]
            combined = "\n".join(batch)
            summary = summarize_chunk(combined)
            next_level.append(summary)
        summaries = next_level
    return summarize_final(summaries[0])
