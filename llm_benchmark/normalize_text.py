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

-------------------------------------------------------------------------------------
ref_criteria_norm = [normalize_text(c) for c in reference_criteria]
llm_criteria_norm = [normalize_text(c) for c in llm_criteria]

ref_emb = MODEL.encode(ref_criteria_norm, convert_to_tensor=True)
llm_emb = MODEL.encode(llm_criteria_norm, convert_to_tensor=True)
