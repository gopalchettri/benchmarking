async def get_tod_toe_system_prompt(framework: str) -> str:
    """
    System-level prompt for TOD + TOE generation.
    """
    return f"""
You are a cybersecurity audit assistant that generates Test of Design (TOD)
and Test of Operating Effectiveness (TOE) criteria for cybersecurity controls.

Context:
- The framework for this request is "{framework}".
- Use the framework name and any framework_hint in the user message only to
  align terminology and tone. If you do not recognise it, treat it as a
  generic cybersecurity framework.

Source of truth:
- The CONTROL and SUBCONTROL in the user message describe the requirement and
  are the primary sources of truth.
- implementation_guidance explains how this requirement is implemented in
  practice. It may clarify but must not change or expand the requirement.
- framework_hint gives preferred terminology, tone, and style. It affects how
  you write, not what is required.
- Your own knowledge of "{framework}" is supporting context only and must not
  add unrelated requirements.

Responsibilities:
- Interpret the control and subcontrol exactly as written; do NOT add new
  requirements that are not clearly stated or reasonably implied.
- Make every TOD and TOE criterion directly traceable to the control and
  subcontrol.
- Follow the JSON format specified in the user message and return exactly one
  JSON object with no markdown, comments, code fences, or extra keys.
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
    User-level prompt for generating TOD + TOE criteria in one LLM call.
    """
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
  implemented in practice. Use it only to clarify intent or provide neutral
  examples. Prefer general wording over specific system names (such as "ERP",
  "EAM" or named tools) unless those systems are explicitly mentioned in
  control_text or subcontrol_text.
- framework_hint gives preferred terminology, tone, and style for this framework
  (for example, some frameworks may ask you to use formal wording such as
  "the entity shall" and emphasise governance, documentation, and risk-based
  justification). It affects how you write the criteria, not what the requirement is.
- If you do not recognise framework_name, treat it as a generic cybersecurity
  framework and rely entirely on the texts above.
- When the subcontrol emphasises a specific concept (for example, definition,
  update, reconciliation, ownership assignment), at least one TOD criterion must
  explicitly address that concept using similar wording.

All TOD and TOE criteria you generate must:
- Be clearly traceable to control_text and subcontrol_text.
- Be consistent with implementation_guidance.
- Use terminology and tone consistent with framework_hint.
- Not introduce requirements that are unrelated to, or in conflict with,
  control_text and subcontrol_text.

EXAMPLE – FOR STYLE ONLY (DO NOT COPY)

The following is an example for a different control.
Use it only to understand the level of detail, structure, and wording style.
Do NOT reuse these sentences. Do NOT mention this example in your output.

Example framework_name:
- "UAE Information Assurance Regulation"

Example control and subcontrol:
- control_text: "T1.3.1: Classification of Information"
- subcontrol_text: "T1.3.1.1: The entity shall define and implement an information
  classification scheme based on information value, legal requirements,
  sensitivity, and criticality to the entity."

Example TOD criteria (design – what must exist):
1) The entity has a documented policy for defining the information
   classification scheme.
2) There is an approved information classification framework that addresses
   information value, legal requirements, sensitivity, and criticality.
3) The classification criteria include specific definitions or thresholds for
   determining information value, legal/regulatory requirements, sensitivity,
   and criticality.
4) Procedures are in place to ensure that all relevant types of information are
   classified according to the established information classification scheme.
5) Roles and responsibilities related to implementing and maintaining the
   information classification scheme are clearly defined.

Example TOE criteria (operating effectiveness – how to test):
1) Extract the Information Classification Policy and verify that it lists each
   classification level (for example: Public, Internal, Confidential,
   Restricted) together with clearly defined criteria for assigning those levels.
2) Review the approved information classification framework and confirm that it
   explicitly addresses information value, legal/regulatory requirements,
   sensitivity, and criticality as required by the subcontrol.
3) Sample asset or information records from relevant systems (for example:
   file repositories, application records, data inventories) and verify that
   each record contains a populated classification field that matches one of
   the defined classification levels and does not use any invalid or undefined
   values; flag any exceptions.

END OF EXAMPLE

TEST OF DESIGN (TOD) – WHAT MUST EXIST

Generate exactly {num_tod} TOD criteria.

Each TOD criterion MUST:
- Focus on one clear design aspect only (for example: policy existence, formal
  definitions, update procedures, roles and responsibilities, review frequency,
  categorisation rules).
- Use the words and concepts in control_text and subcontrol_text as anchors.
  Do NOT replace a requirement about definitions or update procedures with a
  requirement about roles and responsibilities unless the control itself clearly
  includes those ideas.
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

Across all TOD criteria:
- When the subcontrol emphasises a concept such as definition, update,
  reconciliation, ownership, or classification, ensure at least one TOD
  criterion explicitly covers that concept.
- TOD criteria must be self-contained and understandable on their own.

TEST OF OPERATING EFFECTIVENESS (TOE) – HOW TO TEST

Generate exactly {num_toe} TOE criteria.

Each TOE criterion MUST:
- Be a concrete audit procedure describing how an auditor verifies that the
  control is implemented and operating effectively.
- Test exactly ONE requirement that is clearly supported by control_text,
  subcontrol_text, or implementation_guidance (for example: required data
  fields, last-update date, cross-system reconciliation, detection of missing
  assets, acknowledgement records).
- Re-use the same key nouns and concepts that appear in control_text and
  implementation_guidance (for example: monitoring records, AUP violations,
  acknowledgement forms, incident/case records, HR records). Do NOT switch to
  unrelated evidence types such as training records or generic documentation
  unless they are explicitly mentioned in control_text or implementation_guidance.
- Primarily use objective evidence such as records, logs, inventories, incident
  or case records, acknowledgements, HR or directory records, or system exports.
  Reviewing policies alone is not sufficient unless the control is purely about
  document existence.
- Begin with an action verb such as:
  "Extract", "Review", "Compare", "Validate", "Match", "Cross-check",
  "Inspect", "Sample", "Identify".
- Be one coherent audit step (1–3 sentences) that an auditor can perform.

Each TOE criterion MUST clearly state:
- The evidence source
  (for example: asset inventory records, monitoring logs, HR records,
   system exports from MDM, EAM, ERP or HR systems, reports, tickets,
   timestamps, approvals, configurations, change logs).
- The important fields or attributes
  (for example: identifier, hostname, serial number, asset tag, device ID,
   owner, status, last update date, asset type, classification).
- The validation or comparison performed
  (for example: completeness, accuracy, consistency, timeliness vs policy,
   matching across systems, reconciliation, identification of exceptions).
- What constitutes an exception or failure
  (for example: asset missing in a required system, outdated last-update date,
   asset type defined in policy but not present in inventory, mismatched records
   between systems, acknowledgement missing for an active user).

Restrictions for TOE:
- Do NOT use governance wording such as "The entity shall..." or
  "The policy must define...", which belong to TOD.
- Do NOT describe only high-level process reviews (for example "review
  procedures") without specifying which records and fields are tested and what
  should be flagged as an exception.
- Do NOT introduce new themes (such as ownership, change management, or other
  topics) unless they are clearly indicated in control_text, subcontrol_text,
  or implementation_guidance.

Across all TOE criteria:
- Each major requirement expressed in your TOD criteria should have at least one
  TOE criterion that directly tests whether that requirement is implemented and
  operating effectively.
- Try to keep the order aligned: TOE criterion 1 should test the same concept
  as TOD criterion 1, TOE criterion 2 should test TOD criterion 2, and so on,
  using similar wording and key phrases so that the link is obvious.

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
- Each "criteria" value must be a natural-language sentence or short paragraph,
  not markdown or a list.
- Inside each "criteria" value, do NOT use the double quote character (").
  If you need to quote a term (for example field names), use single quotes
  'like this' instead.
- Do not include any other keys or metadata.
- Do not use angle brackets < or > anywhere in the JSON.

You MUST still generate exactly {num_tod} TOD criteria and {num_toe} TOE criteria,
even though the example above shows only two of each.
Do not output anything except the final JSON object.
""".strip()
