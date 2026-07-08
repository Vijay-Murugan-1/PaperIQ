"""
Summary prompt builder utilities for PaperIQ.

This module constructs prompts for generating concise
summaries of research papers using an LLM. 
"""

def build_summary_prompt(document: str) ->str:
    """
    Args: 
        documents (str): 
            complete extracted text of the research paper.
    Returns:
        str:
            Prompt instructing the LLM to summarize the paper. 
    """
    prompt = f"""
You are PaperIQ, an AI-powered research assistant.

Read the following research paper carefully and generate a concise,
well-structured summary.

Your summary should include:

1. The main objective of the paper.
2. The problem being addressed.
3. The proposed methodology.
4. The key findings or contributions.
5. The overall conclusion.  

Do not include unnecessary details, citations, acknowledgements,
or references.

Research Paper:
----------------------------
{document}
----------------------------

Summary:
"""
    return prompt