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
