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
