"""
Embedding generation utilites for PaperIQ

This module provides helper function to provide vector 
embeddings for the research paper chunks using 
Sentence Transformers.These embeddings are later used for
semnatic retrieval in the RAG pipeline.
"""

import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_embedding_model():
    """
    Load and Cache the Sentence Transformer model.

    The model is loaded only once during the application's lifetime
    and reused for the next eembedding requests.

    Returns:
        SentenceTransformer: Initialized sentence embedding model.
    """
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

embedding_model = load_embedding_model()

def generate_embedding(text: str) -> np.ndarray:
    """
    Generate a semantic embedding for a single text chunk.

    Args:
        chunk (str): Input text chunk that is to be converted
        to vector represntation.

    Returns:
        An numpy array representing semantic embeddings.
    
    embedding = embedding_model.encode(text, convert_to_numpy = True)
    return embedding  """


    embedding = embedding_model.encode(
        text,
        convert_to_numpy=True
    )

    return embedding

def generate_embeddings(chunks: list[str]) -> list[np.ndarray]:
    """
    Generates semantic embeddings for multiple text chunks.

    Args:
        chunks(list[str]): List of text chunks extracted 
        form the document.

    Returns:
        List containing one embedding vector for each input chunk.
    """
    embeddings = []

    for chunk in chunks:
        embedding = generate_embedding(chunk)
        embeddings.append(embedding)

    return embeddings
