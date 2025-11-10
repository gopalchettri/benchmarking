import json
import re

def extract_json_from_markdown(text: str):
    """
    Extract JSON from markdown code blocks.
    Handles formats like:
    ```json
    [...]
    ```
    or plain JSON strings
    """
    # Try to extract from markdown code block
    json_pattern = r'```(?:json)?\s*\n(.*?)\n```
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        # Found markdown code block
        json_str = matches.strip()
    else:
        # No code block, assume the whole string is JSON
        json_str = text.strip()
    
    # Parse JSON string to Python object
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nText: {text[:200]}")
