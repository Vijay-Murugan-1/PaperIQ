"""
PDF processing utilities for PaperIQ.

This module handles the extraction of text from uploaded PDF files,
retaining page-level metadata for source citations.
"""

import fitz
def extract_text_from_pdf(uploaded_file: bytes) ->list[dict]:
    """
    Extracts text from a PDF document along with its corresponding page numbers.

    Args:
        pdf_bytes (bytes): The raw byte content of the uploaded PDF.
        
    Returns:
        list[dict]: A list of dictionaries. Each dictionary contains:
            - 'page' (int): The 1-indexed page number.
            - 'text' (str): The extracted text from that page. 
    """
    doc = fitz.open(stream = uploaded_file,filetype="pdf")

    pages_data =[]

    for i,page in enumerate(doc):

        text = page.get_text("text").strip()

        pages_data.append({

        "page": i+1,

        "text": text

        })

    return pages_data
