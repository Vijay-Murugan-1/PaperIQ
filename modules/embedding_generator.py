import streamlit as st
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

embedding_model = load_embedding_model()

def generate_embedding(chunk):
    embedding = embedding_model.encode(chunk, convert_to_numpy = True)
    return embedding

def generate_embeddings(chunks):
    embeddings = []

    for chunk in chunks:
        embedding = generate_embedding(chunk)
        embeddings.append(embedding)

    return embeddings
