def get_tod_toe_user_prompt(
    control: str,
    subcontrol: str,
    framework: str,
    num_tod: int,
    num_toe: int,
) -> str:
    framework_hint = get_framework_hint(framework)

    return f"""
Generate Test of Design (TOD) and Test of Operating Effectiveness (TOE) criteria for the following requirement:

Framework: {framework}
Control: {control}
Subcontrol: {subcontrol}

Apply the following framework-specific guidance where relevant:
{framework_hint}

========================================================
TEST OF DESIGN (TOD) – WHAT MUST EXIST
========================================================
Generate exactly {num_tod} TOD criteria.

Each TOD criterion MUST:
- Be a single-sentence statement describing a required element of the control’s design
  (e.g., policies, procedures, roles, responsibilities, scope, frequency, definitions, documentation).
- Describe what the entity must have in place, not what an auditor does.
- Use formal compliance language (e.g., "Existence of...", "Clear definition of...", "Documented procedures for...").
- Avoid audit or testing verbs such as: verify, extract, inspect, validate, compare, match, check, sample.
- Stay strictly within the intent and scope of the subcontrol; do not introduce unrelated requirements.
- Be specific, measurable, and independent (no vague phrases like "adequate", "appropriate" without context).

Across all TOD criteria together, aim to cover where applicable:
1) A governing policy or documented requirement,
2) Defined scope/coverage and classification or categorization aspects,
3) Roles, responsibilities, and ownership,
4) Processes for review, update, or maintenance of the control.

These TOD criteria should be written in the style of professional cybersecurity audit design criteria,
not tied to any specific individual’s wording.


========================================================
TEST OF OPERATING EFFECTIVENESS (TOE) – HOW TO TEST IT
========================================================
Generate exactly {num_toe} TOE criteria.

Each TOE criterion MUST:
- Describe a concrete audit step that a cybersecurity auditor would perform.
- Start with an action verb such as: "Extract", "Review", "Compare", "Validate", "Cross-check",
  "Match", "Identify", "Inspect", "Sample".
- Clearly state:
  - Which evidence is used (e.g., inventories, logs, policy documents, HR roster, configuration records),
  - What fields or attributes are important (e.g., owner, timestamp, status, identifier),
  - What the auditor is checking (e.g., completeness, accuracy, consistency, timeliness).
- Reflect realistic audit practice: evidence gathering, field checks, sampling, reconciliation between sources, and flagging exceptions.
- Avoid design-language such as "The entity shall…" or "The policy must define…"; that belongs in TOD.
- Be specific and testable so that different auditors would perform essentially the same check.

Across all TOE criteria together, aim to include:
1) Evidence extraction from one or more authoritative systems,
2) Completeness and accuracy checks on key fields,
3) Date/timestamp or review-cycle validation where relevant,
4) Cross-system comparison (e.g., inventory vs HR/AD) when applicable to the subcontrol,
5) Identification of missing, invalid, or outdated records.

These TOE criteria should read like standard audit test procedures that any qualified auditor could follow.


========================================================
OUTPUT FORMAT (STRICT)
========================================================
Return ONLY the JSON object below, with no text before or after:

{{
  "llm_tod_criterias": [
    {{ "id": 1, "criteria": "<TOD criterion>" }},
    {{ "id": 2, "criteria": "<TOD criterion>" }}
    // ... up to {num_tod}
  ],
  "llm_toe_criterias": [
    {{ "id": 1, "criteria": "<TOE criterion>" }},
    {{ "id": 2, "criteria": "<TOE criterion>" }}
    // ... up to {num_toe}
  ]
}}

Rules:
- No markdown, backticks, or commentary.
- Only valid JSON; it must be directly parseable by json.loads() in Python with no changes.
- Each "criteria" value must be a natural-language sentence in double quotes.
- Do not invent new controls; stay within the subcontrol’s intent.
Return only the JSON object.
""".strip()
