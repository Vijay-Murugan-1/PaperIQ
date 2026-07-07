"""
Context builder utilities for PaperIQ

This module constructs a single context string from the
retrieved documnet chunks. The generated context is later
used as input to a Large Language Model (LLM).
"""
import numpy as np

def build_context(chunks: list[str],
                  indices: np.ndarray) ->str:
    """
    Build a context string from the retrieved chunks.

    Args:
        chunks(list[str]): List containing all documnet chunks.

        indices(np.ndarray): Indices of the retrieved chunks 
        returned by FAISS.
    
        Returns:
            str: A formatted context string containing the retrieved
            chunks.
    """
    context = ""

    for rank,chunk_index in enumerate(indices,start=1):
        context += f"Chunk{rank}\n"
        context += "-"*50 + "\n"
        context += chunks[chunk_index]
        context +="\n\n"
    return context