"""
Flashcard generation utilities for PaperIQ.

This module provides helper functions to construct prompts for generating
digital flashcards based on the extracted research paper text. The LLM 
is instructed to return structured JSON for UI rendering.
"""

def build_flashcard(text:str, num_cards: int = 10) ->str:
    """
    Constructs the prompt generte JSON-formatted flashcards from the paper.

    Args:
        tetx (str): The extracted text from the research paper.
        num_cards(int, optional): The number of flashcards to generate. Default to 10.
    
    Returns:
        str: A formatted prompt string instructing the LLM to return a JSON array
        of front/back flashcard objects.  
    """
    prompt = f"""
        You are an expert study assistant. Based on the following research paper text, 
        generate {num_cards} flashcards covering the most important concepts, definitions, 
        methodologies, and findings.

        You MUST output your response strictly as a valid JSON array of objects. 
        Do not include any markdown formatting (like ```json), introduction, or conclusion. 
        Just the raw JSON array.

        Use the exact following structure for each object in the array:
        [
            {{
                "front": "The concept, term, or question (e.g., 'What is RAG?'). Keep it concise.",
                "back": "The definition or answer. Keep it clear, accurate, and easy to memorize."
            }}
        ]

        Text to base the flashcards on:
        {text}
        """
    return prompt.strip()
