async def get_tod_toe_system_prompt(framework: str) -> str:
    """
    Short, production-ready system prompt for TOD + TOE generation.
    """
    return f"""
You are a cybersecurity audit assistant that generates Test of Design (TOD)
and Test of Operating Effectiveness (TOE) criteria.

Context:
- The framework for this request is "{framework}". Use the framework name only to
  align terminology and tone. If you do not recognise it, treat it as a generic
  cybersecurity framework and rely on the information in the user message.

Source of truth:
- CONTROL and SUBCONTROL in the user message describe the requirement and are the
  primary sources of truth.
- implementation_guidance explains how this requirement is implemented in practice.
  It may clarify but must not change or expand the requirement.
- framework_hint gives preferred terminology, tone, and style for this framework.
  It affects how you write, not what is required.
- Your own knowledge of "{framework}" is supporting context only and must not
  introduce unrelated requirements.

Responsibilities:
- Interpret the control and subcontrol exactly as written; do NOT add new
  requirements that are not clearly stated or reasonably implied.
- Make every TOD and TOE criterion directly traceable to the control and
  subcontrol and consistent with the framework context.
- Follow all JSON rules specified in the user message and return exactly one
  JSON object with no markdown, no comments, and no extra keys.
""".strip()


async def get_tod_toe_user_prompt(
    control_id: str,
    control: str,
    subcontrol: str,
    framework: str,
    num_tod: int,
    num_toe: int,
) -> str:
    """
    User prompt text for generating TOD + TOE criteria in one LLM call.
    """
    framework_hint = await get_framework_hint(framework)
    implementation_guidance = await get_control_implementation_guidance(control_id)

    return f"""
Generate Test of Design (TOD) and Test of Operating Effectiveness (TOE) criteria
for the following cybersecurity requirement.

Input data for this request:

- framework_name: {framework}
- control_id: {control_id}
- control_text: {control}
- subcontrol_text: {subcontrol}
- implementation_guidance: {implementation_guidance}
- framework_hint:
{framework_hint}

How to interpret this data:
- Read all fields above in this order:
  framework_name → control_text → subcontrol_text → implementation_guidance → framework_hint.
- control_text and subcontrol_text together describe the requirement and are the primary
  sources of truth.
- implementation_guidance explains how this specific control and subcontrol are expected
  to be implemented in practice. Use it to clarify intent and focus the criteria, but do
  not expand the scope beyond the control_text and subcontrol_text.
- framework_hint provides guidance on preferred terminology, tone, and style for this
  framework (for example, some frameworks may ask you to use formal wording such as
  "the entity shall" and emphasise governance, documentation, and risk-based justification).
  It affects how you write the criteria, not what the requirement is.
- If you do not recognise the framework_name, treat it as a generic cybersecurity
  framework and rely entirely on the texts above.

All TOD and TOE criteria you generate must:
- Be directly traceable to the control_text and subcontrol_text.
- Be consistent with implementation_guidance.
- Use terminology and tone consistent with framework_hint.
- Not introduce requirements that are unrelated to, or in conflict with,
  the control_text and subcontrol_text.

TEST OF DESIGN (TOD) – WHAT MUST EXIST

Generate exactly {num_tod} TOD criteria.

Each TOD criterion MUST:
- Describe what must exist in the design of the control (for example: policies,
  procedures, definitions, roles, responsibilities, documentation, scope,
  frequencies, governance mechanisms).
- Be ONE logical criterion, but may contain multiple sentences if needed for clarity.
- Use formal compliance-style language such as:
  "Existence of documented policy for ...",
  "Clear definition of roles and responsibilities for ...",
  "Documented procedures for ...",
  "Assignment of responsibilities for ...".
- NOT describe audit or testing actions (avoid verbs such as: verify, validate,
  extract, compare, inspect, review, check, sample, reconcile, correlate).
- Stay strictly within the scope and intent of the control_text and subcontrol_text.
- Be measurable, specific, unambiguous, and independent of other criteria.
- Be written as a short paragraph (1–3 sentences), not as a list or bullet points.

TEST OF OPERATING EFFECTIVENESS (TOE) – HOW TO TEST

Generate exactly {num_toe} TOE criteria.

Each TOE criterion MUST:
- Be a concrete audit procedure describing how an auditor verifies that the TOD
  requirements are implemented and operating effectively.
- Begin with an action verb such as:
  Extract, Review, Compare, Validate, Match, Cross-check, Inspect, Sample, Identify.
- Be ONE logical criterion, but may contain multiple sentences if needed
  (for example: evidence step plus validation condition plus exception handling).
- Clearly state:
  - What evidence is used (for example: logs, inventories, HR records, system exports,
    reports, tickets, timestamps, approvals, configurations).
  - Which fields or attributes are important (for example: owner, timestamp, identifier,
    type, status).
  - What validation or comparison the auditor performs (for example: completeness,
    accuracy, consistency, timeliness, correlation).
  - Any cross-system comparison or exception identification when relevant.
- NOT use governance wording such as "The entity shall..." or "The policy must define...",
  which belong to TOD.
- NOT simply restate TOD criteria; TOE criteria must focus on how an auditor tests design
  and operation using evidence.
- Be written as a short paragraph (1–3 sentences), not as a list or bullet points.

All TOD and TOE criteria must be self-contained and understandable on their own:
- Do not refer vaguely to "the above" or "this control".
- Avoid cross-references to other controls unless explicitly required by the subcontrol_text.

STRICT OUTPUT FORMAT – RETURN ONLY THIS JSON

You MUST respond in pure JSON ONLY.
Return ONLY the JSON object described below, with no text before or after.

JSON structure requirements:
- Top-level keys: "control_id", "control", "subcontrol", "framework", "llm_content".
- "llm_content" must contain exactly two keys:
  - "llm_tod_criterias"
  - "llm_toe_criterias"
- "llm_tod_criterias" must be an array of exactly {num_tod} objects.
- "llm_toe_criterias" must be an array of exactly {num_toe} objects.
- In both arrays:
  - Each object must have exactly two keys: "id" (integer) and "criteria" (string).
  - "id" values must be consecutive integers starting at 1 (1, 2, 3, ...).
- Do NOT include any other keys, comments, or metadata.
- Do NOT wrap criteria values in extra quotation marks; the string must not start and end
  with an extra quote character.
- If you need quotation marks inside a criteria, escape them as valid JSON.
- Do NOT use angle brackets < or > in any string values.
- Write each "criteria" value as normal sentences, not as markdown, lists, or bullet points.

Return a JSON object in this shape (text values are only examples; arrays must contain
{num_tod} TOD criteria and {num_toe} TOE criteria):

{{
  "control_id": "{control_id}",
  "control": "{control}",
  "subcontrol": "{subcontrol}",
  "framework": "{framework}",
  "llm_content": {{
    "llm_tod_criterias": [
      {{"id": 1, "criteria": "First TOD criterion in natural language."}},
      {{"id": 2, "criteria": "Second TOD criterion in natural language."}}
    ],
    "llm_toe_criterias": [
      {{"id": 1, "criteria": "First TOE criterion in natural language."}},
      {{"id": 2, "criteria": "Second TOE criterion in natural language."}}
    ]
  }}
}}

Important:
- You MUST still generate exactly {num_tod} TOD criteria and {num_toe} TOE criteria,
  even though the example above shows only two of each.
- Do not output anything except the final JSON object.
""".strip()

