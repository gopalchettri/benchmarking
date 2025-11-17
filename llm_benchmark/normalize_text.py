import re
import string

import re
import string

def normalize_text(text: str) -> str:
    # 1. lower-case
    text = text.lower()

    # 2. remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 3. collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    # 4. remove boilerplate cybersecurity phrases
    boilerplate = [
        "the entity shall",
        "the organization shall",
        "the organisation shall",
        "ensure that",
        "make sure that",
        "must ensure",
    ]
    for phrase in boilerplate:
        text = text.replace(phrase, "")

    # 5. remove safe stopwords
    # These DO NOT remove cybersecurity meaning
    stopwords = {
        "the", "and", "or", "to", "of", "for", "a", "an", 
        "is", "are", "be", "on", "in", "by", "with", 
        "that", "this", "it", "as"
    }

    # remove stopwords ONLY when they are full words
    text = " ".join(
        word for word in text.split()
        if word not in stopwords
    )

    return text


-------------------------------------------------------------------------------------
ref_criteria_norm = [normalize_text(c) for c in reference_criteria]
llm_criteria_norm = [normalize_text(c) for c in llm_criteria]

ref_emb = MODEL.encode(ref_criteria_norm, convert_to_tensor=True)
llm_emb = MODEL.encode(llm_criteria_norm, convert_to_tensor=True)
