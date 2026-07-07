"""
Propmt builder utilities for PpaperIQ.

This module comnstructs prompts that are sent to the 
Large Language Model for question answering.
"""
def build_prompt(context: str,
                 question: str) ->str:

    """
    Args: 
        Context (str): Context built from the retrieved 
        document chunks.

        question (str): User's question.

    Returns: 
        str: Complete prompt for the LLM. 
    """
    prompt =f"""
You are PaperIQ, an AI-powered research assistant.

Anwser the user's question using ONLY provided context.

If the answer cannot be found in the context, say:

"I coudn't find that information in the uploaded paper."

Context:
----------------------
{context}
----------------------

Question:
{question}

Answer
"""
    return prompt
