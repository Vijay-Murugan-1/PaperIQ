"""
vector store utilities for PaperIQ.

This module proviodes helper functions for creating,
populating, and querying a FAISS vector DB.
"""
import faiss
import numpy as np

def create_index(dimension: int) ->faiss.IndexFlatL2:
    """
    create an empty FAISS index.

    Args:
        dimenstion (int): dimension of each embedding vector.

    Returns:
        faiss.IndexFlatL2: An empty FAISS index that is ready to 
        store the embeddings.
    """
    return faiss.IndexFlatL2(dimension)

def add_embeddings(index: faiss.IndexFlatL2,
                   embeddings: list[np.ndarray]) ->None:
    
    """
    Add embedding vectors to FAISS index.

    Args:

        index(faiss.IndexFlatL2): the FAISS index to which the embeddings
        will be added.

        embeddings (list[np.ndarray]): list of embedding vectors generated from
        the document chunks.

    Returns:
        None
    """
    embedding_matrix = np.array(embeddings,dtype =np.float32)
    index.add(embedding_matrix)

def search_index(index: faiss.IndexFlatL2,
                 query_embedding: np.ndarray,
                 top_k: int = 3) ->tuple[np.ndarray,np.ndarray]:
    """
    Search the FAISS index for the most similar embeddings.

    Args:   
        index (faiss.IndexFlatL2): FAISS indesx containing document embeddings.

        query_embedding(np.ndarray): Embedding vector of the user's query.

        top_k (int,optional): Number of nearest neighbors to retrieve. Defaults to 3.

    Returns:
        tuple[np.ndarray,np.ndarray]: A tuple containing distances and indices of the 
        nearest embeddings.
    """

    query_embedding = np.array([query_embedding],dtype=np.float32)
    distances, indices = index.search(query_embedding,top_k)

    return distances[0], indices[0]