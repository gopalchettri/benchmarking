async def get_tod_toe_system_prompt(framework: str) -> str:
    return f"""
You are a cybersecurity audit assistant that generates Test of Design (TOD)
and Test of Operating Effectiveness (TOE) criteria for cybersecurity controls.

Context:
- The framework for this request is "{framework}". Use the framework name and any
  framework_hint in the user message only to align terminology and tone. If you do
  not recognise it, treat it as a generic cybersecurity framework.

Source of truth:
- CONTROL and SUBCONTROL in the user message describe the requirement and are the
  primary sources of truth.
- implementation_guidance explains how this requirement is implemented in practice.
  It may clarify but must not change or expand the requirement.
- framework_hint gives preferred terminology, tone, and style. It affects how you
  write, not what is required.
- Your own knowledge of "{framework}" is supporting context only and must not add
  unrelated requirements.

Responsibilities:
- Interpret the control and subcontrol exactly as written; do NOT add new
  requirements that are not clearly stated or reasonably implied.
- Make every TOD and TOE criterion directly traceable to the control and subcontrol.
- Follow the JSON format specified in the user message and return exactly one JSON
  object with no markdown, comments, code fences, or extra keys.
""".strip()


async def get_tod_toe_user_prompt(
    control_id: str,
    control: str,
    subcontrol: str,
    framework: str,
    num_tod: int,
    num_toe: int,
) -> str:
    framework_hint = await get_framework_hint(framework)
    implementation_guidance = await get_control_implementation_guidance(control_id)

    return f"""
Generate Test of Design (TOD) and Test of Operating Effectiveness (TOE) criteria
for the following cybersecurity requirement.

Input data:
- framework_name: {framework}
- control_id: {control_id}
- control_text: {control}
- subcontrol_text: {subcontrol}
- implementation_guidance: {implementation_guidance}
- framework_hint:
{framework_hint}

How to use this data:
- control_text and subcontrol_text together describe the requirement and are the
  main sources of truth.
- implementation_guidance explains how this specific control and subcontrol are
  implemented in practice. Use it to clarify intent and focus the criteria, but do
  not expand the scope beyond control_text and subcontrol_text.
- framework_hint gives preferred terminology, tone, and style for this framework
  (for example, some frameworks may ask you to use formal wording such as
  "the entity shall" and emphasise governance, documentation, and risk-based
  justification). It affects how you write the criteria, not what the requirement is.
- If you do not recognise framework_name, treat it as a generic cybersecurity
  framework and rely entirely on the texts above.

All TOD and TOE criteria you generate must:
- Be clearly traceable to control_text and subcontrol_text.
- Be consistent with implementation_guidance.
- Use terminology and tone consistent with framework_hint.
- Not introduce requirements that are unrelated to, or in conflict with,
  control_text and subcontrol_text.

TEST OF DESIGN (TOD) – WHAT MUST EXIST

Generate exactly {num_tod} TOD criteria.

Each TOD criterion MUST:
- Describe what must exist in the design of the control (for example: policies,
  procedures, definitions, roles, responsibilities, documentation, scope,
  frequencies, governance mechanisms).
- Be one coherent requirement (1–3 sentences).
- Use governance-style language such as:
  "Existence of documented policy for ...",
  "Clear definition of roles and responsibilities for ...",
  "Documented procedures for ...",
  "Assignment of responsibilities for ...".
- Not describe audit or testing actions (avoid verbs such as: verify, validate,
  extract, compare, inspect, review, check, sample, reconcile, correlate).
- Stay within the scope and intent of control_text and subcontrol_text.
- Be specific, measurable, and unambiguous.

TEST OF OPERATING EFFECTIVENESS (TOE) – HOW TO TEST

Generate exactly {num_toe} TOE criteria.

Each TOE criterion MUST:
- Be a concrete audit procedure describing how an auditor verifies that the TOD
  requirements are implemented and operating effectively.
- Begin with an action verb such as:
  Extract, Review, Compare, Validate, Match, Cross-check, Inspect, Sample, Identify.
- Be one coherent audit step (1–3 sentences) that an auditor can perform.
- Clearly state:
  - The evidence source (for example: logs, inventories, HR records, system exports,
    reports, tickets, timestamps, approvals, configurations).
  - The important fields or attributes (for example: owner, timestamp, identifier,
    type, status).
  - The validation or comparison performed (for example: completeness, accuracy,
    consistency, timeliness, correlation, cross-system reconciliation, exception
    identification).
- Not use governance wording such as "The entity shall..." or "The policy must define...",
  which belong to TOD.
- Focus on how to test, not on restating TOD wording.

General rules:
- TOD and TOE criteria must be self-contained and understandable on their own.
- Do not refer vaguely to "the above" or "this control".
- Do not refer to other controls unless subcontrol_text clearly requires it.

STRICT OUTPUT FORMAT – JSON ONLY

Respond with a single JSON object and nothing else.

Required structure:

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

JSON rules:
- "llm_tod_criterias" must contain exactly {num_tod} objects.
- "llm_toe_criterias" must contain exactly {num_toe} objects.
- In each array, "id" must be consecutive integers starting at 1.
- Each "criteria" value must be a natural-language sentence or short paragraph, not
  markdown or a list.
- Do not include any other keys or metadata.
- Do not wrap criteria values in extra quotes; escape any internal quotation marks
  as valid JSON.
- Do not use angle brackets < or > anywhere in the JSON.

You MUST still generate exactly {num_tod} TOD criteria and {num_toe} TOE criteria,
even though the example above shows only two of each.
Do not output anything except the final JSON object.
""".strip()
