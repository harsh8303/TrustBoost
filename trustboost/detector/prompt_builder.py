"""
trustboost/detector/prompt_builder.py
---------------------------------------
PURPOSE:
    Constructs the detection prompt sent to Gemma 3 4B.

    A well-designed prompt is the most important part of LLM-based detection.
    It tells the model exactly what to look for, what format to use, and what
    to avoid — so we get back reliable, structured JSON every time.

WHY A SEPARATE FILE?
    The prompt is long and careful. Keeping it in its own file means:
    - It is easy to find, read, and improve.
    - The main detector file stays short and focused on logic.
    - You can test prompt-building logic independently.

WHAT THIS FILE DOES NOT DO:
    - Does NOT call Ollama.
    - Does NOT parse responses.
    - Does NOT modify the user's text.
"""


# ── Supported PII Categories ──────────────────────────────────────────────────
# This list is used both in the prompt (so the model knows valid categories)
# and in the validator (so we can reject hallucinated category names).

VALID_ENTITY_TYPES = {
    # Personal identity
    "PERSON",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "LOCATION",
    "DATE_OF_BIRTH",
    # Government and financial
    "AADHAAR",
    "PAN",
    "BANK_ACCOUNT",
    "CREDIT_CARD",
    "IFSC",
    "FINANCIAL_IDENTIFIER",
    # Authentication and security
    "PASSWORD",
    "API_KEY",
    "ACCESS_TOKEN",
    "SECRET",
    "PRIVATE_KEY",
    "CREDENTIAL",
    # Technical and organizational
    "IP_ADDRESS",
    "INTERNAL_URL",
    "DATABASE_CREDENTIAL",
    "SERVER_IDENTIFIER",
    "CONFIDENTIAL_PROJECT",
    "INTERNAL_ORGANIZATION_INFORMATION",
}


def build_detection_prompt(text: str) -> str:
    """
    Constructs the full prompt to send to Gemma 3 4B for PII detection.

    WHY THIS FUNCTION EXISTS:
        The prompt is the "instruction manual" we hand to the LLM. Getting it
        right is critical — a vague prompt leads to missed detections or false
        positives. A good prompt leads to precise, structured, trustworthy output.

    DESIGN PRINCIPLES USED HERE:
        1. Clear role assignment — tell the model exactly what it is doing.
        2. Explicit output format — show it exactly what JSON to produce.
        3. Clear examples — show correct and incorrect detections.
        4. Explicit restrictions — tell it what NOT to do (no rewriting, etc.).
        5. Context requirement — remind it that context determines sensitivity.

    PARAMETERS:
        text (str): The raw user-provided text to analyze.

    RETURNS:
        str: The complete prompt string to send to Gemma 3 4B.
    """

    # We build the list of valid categories as a readable string for the prompt.
    categories_str = "\n".join(f"  - {cat}" for cat in sorted(VALID_ENTITY_TYPES))

    prompt = f"""You are a Personally Identifiable Information (PII) detection system.

Your task is to analyze the TEXT provided below and identify any sensitive personal information.

## IMPORTANT RULES

1. Return ONLY a valid JSON object. No explanations. No markdown. No code blocks.
2. Do NOT modify, rewrite, summarize, or paraphrase the text.
3. Entity "text" values must be copied EXACTLY as they appear in the original text.
4. The "start" position is the character index where the entity begins (0-indexed, inclusive).
5. The "end" position is the character index where the entity ends (exclusive).
   So: original_text[start:end] must equal the entity "text" exactly.
6. Only detect information that is genuinely sensitive given its context.
7. Do NOT detect generic numbers, common dates, company names, or public locations unless they are clearly sensitive in context.
8. If NO sensitive information is found, return an empty entities list.

## VALID ENTITY TYPES

Only use these exact type names:
{categories_str}

## CONTEXT RULES

- A person's name is PII. Generic role titles like "manager" or "user" are NOT.
- College assignments, coursework, academic project descriptions, and technical disciplines (e.g. "machine learning project") are NOT sensitive or persons.
- A specific email address is PII. Generic words like "email" are NOT.
- A phone number is PII. The word "number" alone is NOT.
- A password or API key is always PII.
- An IP address is PII. The phrase "IP address" alone is NOT.
- A project name is CONFIDENTIAL_PROJECT only if the surrounding text indicates it is confidential, internal, or secret.
- Marks, scores, and academic results are NOT PII unless linked to a person's name.
- General meeting times are NOT PII.
- Public company names and general country names (e.g. "India") are NOT PII unless combined with confidential context or private residential addresses.
- An Aadhaar number, PAN card, or bank account number is always PII.

## REQUIRED JSON FORMAT

{{
  "entities": [
    {{
      "text": "<exact text from original>",
      "type": "<VALID_ENTITY_TYPE>",
      "start": <integer start position>,
      "end": <integer end position>,
      "reason": "<one short sentence explaining why this is sensitive>"
    }}
  ]
}}

If no PII is found:
{{
  "entities": []
}}

## EXAMPLES

Example 1:
Input: "Rahul Sharma's email is rahul@example.com."
Output:
{{
  "entities": [
    {{"text": "Rahul Sharma", "type": "PERSON", "start": 0, "end": 12, "reason": "Full name of an individual"}},
    {{"text": "rahul@example.com", "type": "EMAIL", "start": 24, "end": 41, "reason": "Personal email address"}}
  ]
}}

Example 2:
Input: "The meeting is scheduled for 10 AM tomorrow."
Output:
{{
  "entities": []
}}

Example 3:
Input: "My password is Admin@123 and my IP is 192.168.1.10."
Output:
{{
  "entities": [
    {{"text": "Admin@123", "type": "PASSWORD", "start": 15, "end": 24, "reason": "Authentication credential"}},
    {{"text": "192.168.1.10", "type": "IP_ADDRESS", "start": 37, "end": 49, "reason": "Internal IP address"}}
  ]
}}

Example 4:
Input: "I scored 95 marks in mathematics."
Output:
{{
  "entities": []
}}

Example 5:
Input: "We are working on a confidential project called Project Falcon."
Output:
{{
  "entities": [
    {{"text": "Project Falcon", "type": "CONFIDENTIAL_PROJECT", "start": 48, "end": 62, "reason": "Internal confidential project name"}}
  ]
}}

## TEXT TO ANALYZE

{text}

## YOUR RESPONSE (JSON only):"""

    return prompt
