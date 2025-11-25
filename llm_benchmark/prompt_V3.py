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

Control metadata:
- framework: {framework}
- control_id: {control_id}
- control: {control}
- subcontrol: {subcontrol}
- implementation_guidance: {implementation_guidance}

Framework-specific guidance (supporting reference):
{framework_hint}

Follow this understanding sequence BEFORE generating any criteria:
1) First, recall your knowledge of the "{framework}" framework and how controls and subcontrols
   are typically structured and tested.
2) Then, carefully read and internalise the framework-specific guidance text shown above
   (framework_hint) to align terminology and typical expectations.
3) Next, read the control and subcontrol in detail and understand their exact objective and scope.
4) Finally, read the implementation_guidance and use it to clarify what the organisation is expected
   to do to meet the subcontrol.
5) All TOD and TOE criteria you generate must:
   - Be directly traceable to the subcontrol.
   - Be consistent with the framework and framework_hint.
   - Respect the practical intent of the implementation_guidance.
   - Not introduce requirements that are unrelated to, or in conflict with, the subcontrol.

TEST OF DESIGN (TOD) – WHAT MUST EXIST

Generate exactly {num_tod} TOD criteria.

Each TOD criterion MUST:
- Describe what must exist in the design of the control (for example: policies, procedures,
  definitions, roles, responsibilities, documentation, scope, frequency, governance mechanisms).
- Be ONE logical criterion, but may contain multiple sentences when needed to express a complete
  requirement clearly.
- Use formal compliance language such as:
  "Existence of documented policy for ...",
  "Clear definition of roles and responsibilities for ...",
  "Documented procedures for ...",
  "Assignment of responsibilities for ...".
- Not describe audit or testing actions. Avoid verbs such as: verify, validate, extract, compare,
  inspect, review, check, sample, reconcile, correlate.
- Stay strictly within the scope and intent of the subcontrol; do not introduce unrelated topics.
- Be measurable, specific, unambiguous, and independent of other criteria.
- Be written as a short paragraph (1–3 sentences), not a list or bullet points.

Collectively, TOD criteria should (where applicable):
1) Establish the presence of required policy, procedure, or documentation.
2) Define the scope, categorisation, or coverage of the control.
3) Clarify roles and responsibilities.
4) Define update, maintenance, review, or lifecycle expectations.

TEST OF OPERATING EFFECTIVENESS (TOE) – HOW TO TEST

Generate exactly {num_toe} TOE criteria.

Each TOE criterion MUST:
- Be a concrete audit procedure describing how an auditor verifies that the TOD requirements are
  implemented and operating effectively in practice.
- Begin with an action verb such as: Extract, Review, Compare, Validate, Match, Cross-check,
  Inspect, Sample, Identify.
- Be ONE logical criterion, but may contain multiple sentences if needed
  (for example: evidence step plus validation condition plus exception handling).
- Clearly state:
  - What evidence is used (for example: logs, inventories, HR records, system exports, reports,
    tickets, timestamps, approvals, configurations).
  - Which fields or attributes are important (for example: owner, timestamp, identifier, type, status).
  - What validation or comparison the auditor performs (for example: completeness, accuracy,
    consistency, timeliness, correlation).
  - Any cross-system comparison or exception identification when relevant.
- Not use governance wording such as "The entity shall..." or "The policy must define...",
  which belong to TOD.
- Not simply restate TOD criteria; TOE criteria must focus on how an auditor tests design
  and operation using evidence.
- Be written as a short paragraph (1–3 sentences), not a list or bullet points.

Collectively, TOE criteria must include:
1) Evidence extraction from authoritative systems.
2) Field-level validation and completeness checks.
3) Date/timestamp review when relevant.
4) Cross-system reconciliation where applicable.
5) Identification of missing, outdated, or inconsistent records.

All TOD and TOE criteria must be self-contained and understandable on their own:
- Do not refer vaguely to "the above" or "this control".
- Avoid cross-references to other controls unless the subcontrol explicitly requires them.

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
- Do not include any other keys, comments, or metadata.
- Do not wrap criteria values in extra quotation marks; the string must not start and end
  with a literal quote character.
- If you need quotation marks inside a criteria, escape them as valid JSON.
- Do not use angle brackets < or > in any string values.
- Write each "criteria" value as normal sentences, not as markdown, lists, or bullet points.

Return a JSON object in this shape (the text values are only examples; the actual
arrays must contain {num_tod} TOD criteria and {num_toe} TOE criteria):

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
