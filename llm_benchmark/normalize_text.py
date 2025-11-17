import re
import string

def normalize_text(text: str) -> str:
    # lower-case
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # optional: remove boilerplate phrases common across all criteria
    boilerplate = [
        "the entity shall",
        "the organization shall",
        "the organisation shall",
        "ensure that",
    ]
    for phrase in boilerplate:
        text = text.replace(phrase, "")
    return text
