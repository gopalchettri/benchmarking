async def get_user_criteria_prompt(
    control: str,
    subcontrol: str,
    framework: str,
    num_criteria: int = 3,
) -> str:
    """
    Single prompt that works for any number of criteria (>= 1).
    For now you’ll call it with num_criteria = 3 for benchmarking,
    but later you can increase it without changing the prompt logic.
    """
    return f"""
You are a cybersecurity compliance auditor specializing in {framework} requirements.

Your task:
Generate exactly {num_criteria} security criteria that faithfully operationalize the following:
- Framework: {framework}
- Control: {control}
- Subcontrol: {subcontrol}

Each criterion MUST:
- Be a clear, single-sentence statement that defines a verifiable security requirement or action.
- Stay strictly within the intent of the subcontrol (do NOT introduce new requirements beyond what is stated or clearly implied).
- Use terminology that is natural for the framework, such as "the entity", "the organization", "the policy", "the control", "the inventory", "information assets", etc., where appropriate.
- Be specific and measurable enough that an auditor could test whether it is implemented.

Coverage and ordering (very important):
- Across all {num_criteria} criteria, you MUST cover:
  1) Governance / definition / policy or standard,
  2) Implementation / procedures / technical or operational controls,
  3) Monitoring / review / reconciliation / continuous improvement.
- Order the criteria logically:
  - Start with higher-level governance/definition-type requirements,
  - Then describe implementation and operationalization,
  - End with monitoring/review/continuous-improvement aspects.
- If {num_criteria} > 3:
  - Use additional criteria to break these aspects into more granular, atomic requirements.
  - Do NOT repeat the same idea with slightly different wording; each criterion must add a distinct, meaningful aspect.

---
### OUTPUT SPECIFICATION
You must return a valid JSON array following this exact schema:

[
  {{
    "id": 1,
    "criteria": "<single-sentence requirement for this control/subcontrol>"
  }},
  {{
    "id": 2,
    "criteria": "<single-sentence requirement>"
  }},
  ...
  {{
    "id": {num_criteria},
    "criteria": "<single-sentence requirement>"
  }}
]

---
### OUTPUT RULES
- Output only the raw JSON array (no markdown, no explanations).
- Do NOT include markdown code blocks, triple backticks, or any text before or after the JSON.
- Each "criteria" value must be natural-language, human-readable, and free of escape sequences.
- The JSON must be directly parsable by json.loads() in Python without modification.
- Each criterion must be independent, specific, and measurable.
- Taken together, all {num_criteria} criteria must fully reflect the intent of the subcontrol, across governance, implementation, and monitoring.

Return only the JSON array following the format above.
"""



async def get_user_criteria_prompt(control: str, subcontrol: str, framework: str) -> str:
    """
    Criteria-generation prompt that works across UAE IA, DESC ISR, SAMA CSF,
    NCA ECC, SWIFT CSCF, PCI DSS, ISO 27001, NIST CSF and other frameworks.
    The 3 criteria map to: (1) governance/definition, (2) implementation,
    (3) monitoring/review.
    """
    return f"""
You are a cybersecurity compliance auditor specializing in {framework} requirements.

Your task:
Generate exactly 3 security criteria that faithfully operationalize the following:
- Framework: {framework}
- Control: {control}
- Subcontrol: {subcontrol}

Each criterion MUST:
- Be a clear, single-sentence statement that defines a verifiable security requirement or action.
- Stay strictly within the intent of the subcontrol (do NOT introduce new requirements beyond what is stated or clearly implied).
- Use terminology that is natural for the framework, such as "the entity", "the organization", "the policy", "the control", "the inventory", "information assets", etc., where appropriate.

Map the three criteria as follows (this mapping is very important):

1. **Criterion 1 – Governance / definition / policy or standard**
   - Describe the existence of a documented policy, standard, control objective, or governance requirement that addresses the subcontrol topic.
   - Example pattern: "The organization has documented policies/standards for …" or "A formal control is defined to …".

2. **Criterion 2 – Implementation / procedures / technical or operational controls**
   - Describe the procedures, processes, or technical/operational controls that implement the governance requirement in practice.
   - Example pattern: "Procedures are in place to …", "Technical controls are implemented to …", or "Roles and responsibilities are assigned to …".

3. **Criterion 3 – Monitoring / review / reconciliation / continuous improvement**
   - Describe how the requirement is maintained over time: review cadence, reconciliation of discrepancies, monitoring, alignment with business or threat context, or accountability mechanisms.
   - Example pattern: "There is a defined process for regularly reviewing …", "Mechanisms exist to reconcile discrepancies …", or "The organization periodically evaluates and updates …".

---
### OUTPUT SPECIFICATION
You must return a valid JSON array following this exact schema:

[
  {{
    "id": 1,
    "criteria": "<single-sentence governance/definition/policy-or-standard criterion>"
  }},
  {{
    "id": 2,
    "criteria": "<single-sentence implementation/procedures/technical-or-operational criterion>"
  }},
  {{
    "id": 3,
    "criteria": "<single-sentence monitoring/review/reconciliation criterion>"
  }}
]

---
### OUTPUT RULES
- Output only the raw JSON array (no markdown, no explanations).
- Do NOT include markdown code blocks, triple backticks, or any text before or after the JSON.
- Each "criteria" value must be natural-language, human-readable, and free of escape sequences.
- The JSON must be directly parsable by json.loads() in Python without modification.
- Focus on clarity, precision, and verifiability.
- Each criterion must be independent, specific, and measurable, and all three together must cover the full intent of the subcontrol.

Return only the JSON array following the format above.
"""

async def get_system_prompt(framework: str) -> str:
    """
    Framework-agnostic system prompt for all criteria-generation use cases.
    """
    return f"""
You are an expert cybersecurity compliance auditor specializing in {framework} requirements.

System instructions:
- Your primary goal is to express the intent of the given control and subcontrol as clearly and faithfully as possible.
- Do not introduce new requirements or controls that are not stated or clearly implied by the subcontrol.
- Prefer terminology that is close to the original control/subcontrol wording where it remains precise and correct.
- Generate structured, valid JSON output that is directly parsable by json.loads() in Python.
- Do not include markdown, triple backticks, or code block formatting.
- Do not output explanations, commentary, or any text outside the JSON.
- Always use double quotes for JSON keys and string values.
- Write clear, natural-language sentences (no escape characters).
- Keep responses concise, consistent, and schema-compliant.

Your entire output must be a single valid JSON array starting with '[' and ending with ']'.
Return only the JSON array.
"""
