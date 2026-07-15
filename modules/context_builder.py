"""
Context builder utilities for PaperIQ

This module constructs a single context string from the
retrieved documnet chunks. The generated context is later
used as input to a Large Language Model (LLM).
"""
import numpy as np

def build_context(chunks: list[dict],
                  indices: np.ndarray) ->str:
    """
    Build a context string from the retrieved chunks.

    Args:
        chunks(list[dict]): List containing all documnet chunks, where each chunk is
         a dictionary containing 'page' and 'text'.

        indices(np.ndarray): Indices of the retrieved chunks 
        returned by FAISS.
    
        Returns:
            str: A formatted context string containing the retrieved
            chunks.
    """
    context = ""

    for rank,chunk_index in enumerate(indices,start=1):
        if chunk_index < len(chunks):
          chunk = chunks[chunk_index]  
          page_num = chunk["page"]
          chunk_text = chunk["text"]
          context += f"Chunk{rank} (Source: Page {page_num})\n"
          context += "-"*50 + "\n"
          context += chunk_text
          context +="\n\n"
    return context