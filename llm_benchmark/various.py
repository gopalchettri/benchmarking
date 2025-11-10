import json
import re

def extract_json_from_markdown(text: str):
    """Extract JSON from markdown code block or plain string and parse as Python object."""
    # Remove leading/trailing triple quotes and whitespace
    text = text.strip().strip('`"\'')

    # Try markdown extraction first
    match = re.search(r'``````', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # Strip out any other text before/after the JSON
        json_str = text
        # If text still starts with 'json\n', remove it
        if json_str.lower().startswith("json\n"):
            json_str = json_str[5:].strip()

    # Now, try to parse as JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nRaw: {text[:180]}")


async def get_criteria_prompt(control: str, subcontrol: str, framework: str) -> str:
    """
    Universal user prompt for benchmarking all models.
    Optimized for clarity, consistency, and fair evaluation.
    """
    return (
        f"Framework: {framework}\n"
        f"Control: {control}\n"
        f"Subcontrol: {subcontrol}\n\n"
        f"Task: Generate exactly 3 compliance evaluation criteria.\n\n"
        f"Each criterion must:\n"
        f"1. Be clear and measurable\n"
        f"2. Focus on policy-level compliance (not technical implementation details)\n"
        f"3. Include a brief title (3-5 words)\n"
        f"4. State the core requirement\n"
        f"5. Specify design-level verification (what to check in documentation/plans)\n"
        f"6. Specify implementation-level verification (what to check in production systems)\n\n"
        f"Combine all parts into ONE continuous sentence per criterion.\n\n"
        f"EXAMPLE OUTPUT:\n"
        f"[\n"
        f"  {{\n"
        f"    \"id\": 1,\n"
        f"    \"criteria\": \"Access Control Matrix - Define comprehensive role-based permissions for all system resources. Verify that design documentation includes detailed permission matrix with roles...\"\n"
        f"  }},\n"
        f"  {{\n"
        f"    \"id\": 2,\n"
        f"    \"criteria\": \"Multi-factor Authentication Policy - Implement MFA for all privileged accounts and remote access. Confirm that security architecture specifies MFA mechanisms and enrollment...\"\n"
        f"  }},\n"
        f"  {{\n"
        f"    \"id\": 3,\n"
        f"    \"criteria\": \"Security Audit Logging - Maintain comprehensive tamper-evident logs for all security events. Review logging strategy documentation covering log sources and retention requirements...\"\n"
        f"  }}\n"
        f"]\n\n"
        f"CRITICAL FORMATTING RULES:\n"
        f"- Output ONLY the raw JSON array\n"
        f"- Do NOT use markdown code blocks\n"
        f"- Do NOT add triple backticks (```
        f"- Do NOT add ```json or any formatting markers\n"
        f"- Do NOT wrap in markdown\n"
        f"- Do NOT add explanations before or after the JSON\n"
        f"- Write natural sentences without escape characters\n"
        f"- Your entire response must be parsable by json.loads() in Python\n\n"
        f"Generate 3 criteria for {framework} {control} {subcontrol}.\n"
        f"Return only the JSON array as shown in the example above."
    )

# src/core/utils.py (or wherever you keep utility functions)

import json
import re

# src/core/utils.py

import json
import re

def extract_json_from_markdown(text: str):
    """
    Extracts and parses JSON from LLM responses wrapped in markdown or quotes.
    Handles:
    - Markdown code blocks with backticks (```json ... ```
    - Triple single quotes ('''json ... ''')
    - Triple double quotes ("""json ... """)
    - Plain JSON strings
    - Already-parsed Python objects
    """
    # If already parsed, return as-is
    if isinstance(text, (list, dict)):
        return text
    
    if not isinstance(text, str):
        raise ValueError(f"Expected string, list, or dict, got {type(text)}")
    
    text = text.strip()
    
    # Try to extract from code block (handles ```, ''', or """)
    # Pattern matches: `````` or '''json\n...\n''' or similar
    patterns = [
        r'``````',      # Triple backticks
        r"'''(?:json)?\s*\n(.*?)\n'''",      # Triple single quotes
        r'"""(?:json)?\s*\n(.*?)\n"""',      # Triple double quotes
    ]
    
    json_str = None
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            break
    
    # If no code block found, use the whole text
    if json_str is None:
        json_str = text
    
    # Clean up any remaining artifacts
    json_str = json_str.strip().strip('`').strip("'").strip('"')
    
    # Parse JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to parse JSON from LLM output.\n"
            f"Error: {e}\n"
            f"First 300 chars: {text[:300]}"
        )

async def get_criteria_prompt(control: str, subcontrol: str, framework: str) -> str:
    """
    Generates a production-grade criteria-generation prompt
    with deterministic, valid JSON output structure.
    """
    return f"""
You are a cybersecurity compliance expert specializing in {framework} standards.

Your task:
Generate exactly 3 distinct, concise, and actionable security criteria related to:
- Framework: {framework}
- Control: {control}
- Subcontrol: {subcontrol}

Each criterion should be a clear, single-sentence statement that defines a verifiable security requirement or action item. Avoid redundancy and ensure criteria are self-contained.

---

### OUTPUT SPECIFICATION
You must return a **valid JSON array** following this exact schema:

[
  {{
    "id": 1,
    "criteria": "<clear, complete, single-sentence security criterion>"
  }},
  {{
    "id": 2,
    "criteria": "<next criterion>"
  }},
  {{
    "id": 3,
    "criteria": "<next criterion>"
  }}
]

---

### OUTPUT RULES
- Output **only** the raw JSON array (no markdown, no explanations).
- **Do not** include markdown code blocks, triple backticks, or quotes around the JSON.
- **Do not** include any text before or after the JSON.
- Each `criteria` value must be natural language, human-readable, and free of escape sequences (no \\n or \\ characters).
- The JSON must be directly parsable by `json.loads()` in Python without modification.
- Focus on clarity, precision, and verifiability.
- Avoid vague terms like "ensure", "should", or "as appropriate".
- Each criterion must be **independent, specific, and measurable**.

---

### EXAMPLE OUTPUT
[
  {{
    "id": 1,
    "criteria": "Access Control Matrix - Define and enforce role-based permissions for all user roles across systems."
  }},
  {{
    "id": 2,
    "criteria": "Multi-Factor Authentication Policy - Enforce MFA for privileged accounts and remote access."
  }},
  {{
    "id": 3,
    "criteria": "Security Audit Logging - Maintain tamper-evident logs for all system security events."
  }}
]

Return only the JSON array following the format above.
"""
async def get_system_prompt(framework: str) -> str:
    """
    Returns a universal system prompt ensuring consistent, schema-adhering JSON output.
    """
    return f"""
You are an expert cybersecurity compliance auditor specializing in {framework} standards.

System instructions:
- Generate structured, valid JSON output that is directly parsable by `json.loads()` in Python.
- Do not include markdown, triple backticks, or code block formatting.
- Do not output explanations, commentary, or any text outside the JSON.
- Always use double quotes for JSON keys and string values.
- Write clear, natural language sentences (no escape characters).
- Keep responses concise, balanced, and schema-compliant.

Your entire output must be a single valid JSON array starting with `[` and ending with `]`.
Return only the JSON array.
"""
import re
import json

def extract_and_validate_json(raw_text: str):
    """
    Cleans and extracts valid JSON array from LLM output text like:
    ```json
    [ { "id": 1, "criteria": "..." }, ... ]
    ```
    Returns validated list of dicts.
    """

    # Step 1: Remove markdown and formatting noise
    cleaned = re.sub(r"```.*?```", "", raw_text, flags=re.DOTALL)  # Remove markdown blocks
    cleaned = re.sub(r"```json", "", cleaned)
    cleaned = cleaned.replace("\\n", "").replace("\\", "")
    cleaned = cleaned.replace("\n", "").replace("\r", "")
    cleaned = cleaned.strip()

    # Step 2: Extract the first valid JSON array from the text
    match = re.search(r"\[.*\]", cleaned)
    if not match:
        raise ValueError("No JSON array found in the text.")
    
    json_str = match.group(0)

    # Step 3: Parse the JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    # Step 4: Validate structure
    validated = []
    for item in data:
        if not isinstance(item, dict):
            continue
        id_val = item.get("id")
        criteria_val = item.get("criteria")

        if isinstance(id_val, int) and isinstance(criteria_val, str) and criteria_val.strip():
            validated.append({
                "id": id_val,
                "criteria": criteria_val.strip()
            })

    if not validated:
        raise ValueError("No valid criteria objects found.")

    return validated


# Example Usage
if __name__ == "__main__":
    messy_text = r"""
    ```json
    [
      { "id":1, "criteria":"Identify Stakeholders - The entity should identify all stakeholders relevant to its information security. Verify through design documentation that a comprehensive list of relevant stakeholders is defined, detailing both internal and external parties." },
      { "id":2, "criteria":"Stakeholder Needs Analysis - The entity must analyze and document the information security needs and concerns of stakeholders. Confirm that plans include a structured process to capture stakeholder input and align it with security priorities." },
      { "id":3, "criteria":"Risk Impact on Stakeholders - Assess risks associated with stakeholders’ information security interests. Ensure that risk assessment reports identify potential impacts on stakeholders and include mitigation strategies." }
    ]
    ```
    """

    print(extract_and_validate_json(messy_text))
