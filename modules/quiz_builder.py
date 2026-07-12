"""
Quiz generation utilites for PaperIQ.

This module provides helper functions to construct prompts for generating 
multiple-choice quizzes based on the extracted research paper text. These
prompts force the LLM to return structured JSON for UI rendering.
"""
def build_quiz_prompt(text: str, num_questions: int = 5) ->str:
    """
    Constructs the prompt to generate a JSON-formatted quiz fromt the paper.

    Args:
        text (str) :The extracted text from the research paper to base the quiz on.
        num_questions (int, optional): The number of multiple-choice questions 
        to generate. Defaults to 5.

    Returns:
        str: A formatted prompt string instructing the LLM to return a JSON array 
        of quiz questions.
    """
    prompt = f"""
    You are an expert educational assistant. Based on the following research paper text, 
    generate a multiple-choice quiz with {num_questions} questions to test the reader's 
    understanding of the core concepts, methodologies, or findings.

    You MUST output your response strictly as a valid JSON array of objects. 
    Do not include any markdown formatting (like ```json), introduction, or conclusion. 
    Just the raw JSON array.

    Use the exact following structure for each object in the array:
    [
        {{
            "question": "The question text",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "The exact string of the correct option",
            "explanation": "A brief explanation of why this is the correct answer based on the text."
        }}
    ]

    Text to base the quiz on:
    {text}
    """
    return prompt.strip()