"""
Text chunking utilities for PaperIQ.

This module splits extracted PDF pages into token-based, model-friendly chunks
using the Hugging Face transformers library, while preserving page-level metadata 
for source citations.
"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

def create_chunks(pages_data: list[dict],chunk_size = 512,overlap = 50) ->list[dict]:
    """
    Splits page text into token-based chunks and attaches page numbers for citation.

    Args:
        pages_data (list[dict]): A list of dictionaries containing 'page' and 'text'.
        chunk_size (int, optional): The maximum number of tokens per chunk. Defaults to 512.
        overlap (int, optional): The number of overlapping tokens between chunks. Defaults to 50.

    Returns:
        list[dict]: A list of chunk dictionaries, each containing:
            - 'page' (int): The page number this chunk belongs to.
            - 'text' (str): The string decoded from the token chunk.
    """
    chunks = []
    stride = chunk_size - overlap
    for page in pages_data:
        text =page["text"]
        page_num = page["page"]
        tokens= tokenizer.encode(text,
                             add_special_tokens = False,
                             truncation = False)
    for start in range(0,len(tokens),stride):
        chunk_tokens = tokens[start : start + chunk_size] 
    chunks = []
    stride = chunk_size - overlap

    for start in range (0,len(tokens),stride):

        chunk_tokens = tokens[start : start + chunk_size]

        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens = True)
        chunks.append({ "page": page_num,
            "text":chunk_text})

    return chunks