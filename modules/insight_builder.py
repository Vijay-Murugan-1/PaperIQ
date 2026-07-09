"""
Key Insight prompt builder utilities for PaperIQ.

This module constructs prompts for extracting the
most important insights from a research paper.

"""

def build_insight_prompt(document: str) ->str:
    """
    Args: 
        document (str): 
            complete extracted text of the research paper.
    Returns:
        str: 
            Prompt instructing the LLM to generate key insights.
    """
    prompt = f"""
You are PaperIQ, an AI-powered research assistant.

Read the following research paper carefully and identify the
5 most important insights.

For each insight:
-Give a short title.
-Explain it in 2-3 sentences.
-Keep the explanation simple and concise.

Do not include refernces, acknowlwdgements, or citations.

Research Paper:
-------------------------
{document}
-------------------------

Key Insights:
"""
    return prompt