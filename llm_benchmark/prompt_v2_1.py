"""
prompts_tod_toe.py

Prompt builders for generating Test of Design (TOD) and
Test of Operating Effectiveness (TOE) criteria for cybersecurity frameworks.

The LLM is instructed to return JSON in this format:

{
  "control_id": "...",
  "control": "...",
  "subcontrol": "...",
  "framework": "...",
  "llm_content": {
    "llm_tod_criterias": [
      { "id": 1, "criteria": "..." },
      ...
    ],
    "llm_toe_criterias": [
      { "id": 1, "criteria": "..." },
      ...
    ]
  }
}

Typical usage:

from prompts_tod_toe import build_tod_toe_messages
import json

messages = build_tod_toe_messages(
    control_id="T1.2.1.2",
    control="T1.2.1.2: Inventory of Information and other Associated Assets",
    subcontrol="11.2.1.2: The entity shall identify and maintain an up-to-date inventory of information assets within the entity.",
    framework="UAE Information Assurance Regulation",
    num_tod=3,
    num_toe=3,
)

response = client.chat.completions.create(
    model="phi-4",
    messages=messages,
    temperature=0.0,
    max_tokens=700,
    # do_sample = false in your backend
)

raw = response.choices[0].message.content
data = json.loads(raw)  # data["llm_content"]["llm_tod_criterias"], etc.
"""

from typing import List, Dict


# ---------------------------------------------------------------------
# Framework-specific tone / style hints
# ---------------------------------------------------------------------

FRAMEWORK_HINT_MAP: Dict[str, str] = {
    "UAE Information Assurance Regulation": """
- Use formal wording such as “the entity shall” or “the organization shall” where appropriate.
- Reflect the formal, regulatory tone typically used in UAE IA controls.
- Emphasize governance, documentation, and risk-based justifications where needed.
""",
    "UAE IA": """
- Use formal wording such as “the entity shall” or “the organization shall” where appropriate.
- Reflect the formal, regulatory tone typically used in UAE IA controls.
""",
    "SAMA CSF": """
- Use terminology suitable for banking and financial institutions (e.g., “the bank”, “financial institution”, “customer information”).
- Maintain a formal, risk-based tone aligned with the SAMA Cyber Security Framework.
""",
    "NIST CSF": """
- Use risk-based language aligned with NIST Cybersecurity Framework functions (Identify, Protect, Detect, Respond, Recover).
- Emphasize roles, responsibilities, and documented processes.
""",
    "ISO 27001": """
- Align with ISO 27001 Annex A control style.
- Use formal, management-system oriented language (policies, procedures, responsibilities, monitoring).
""",
    # Default fallback if framework not explicitly listed
    "_default_": """
- Use formal, professional cybersecurity and risk-management terminology.
- Maintain a governance, risk, and compliance oriented tone.
"""
}


def get_framework_hint(framework: str) -> str:
    """
    Resolve a framework-specific hint, falling back to a generic default.
    """
    key = (framework or "").strip()
    return FRAMEWORK_HINT_MAP.get(key, FRAMEWORK_HINT_MAP["_default_"])


# ---------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------

def get_tod_toe_system_prompt(framework: str) -> str:
    """
    System prompt for TOD + TOE generation.
    This establishes the role and global behavior for the LLM.
    """
    return f"""
You are a senior cybersecurity auditor specializing in {framework} and global audit practices.

You generate two outputs:
1) Test of Design (TOD) criteria – governance-level requirements describing what must exist.
2) Test of Operating Effectiveness (TOE) criteria – evidence-based audit procedures describing how to test.

Your responsibilities:
- Interpret controls and subcontrols exactly as written.
- Do NOT add new requirements that are not clearly stated or reasonably implied.
- Generate criteria that match the tone, structure, and terminology of {framework}.
- Ensure all output follows strict JSON formatting and can be parsed by json.loads() without modification.
- Keep content concise, auditable, consistent, and domain-accurate.

Your entire output must be only the JSON object. No markdown, no commentary, no code fences.
""".strip()


# ---------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------

def get_tod_toe_user_prompt(
    control_id: str,
    control: str,
    subcontrol: str,
    framework: str,
    num_tod: int,
    num_toe: int,
) -> str:
    """
    User prompt text for generating TOD + TOE criteria in one LLM call.

    - Includes control_id, control, subcontrol, framework in the output.
    - Supports multi-sentence criteria (as one logical criterion).
    - Enforces separation between TOD (design) and TOE (testing).
    """
    framework_hint = get_framework_hint(framework)

    return f"""
Generate Test of Design (TOD) and Test of Operating Effectiveness (TOE) criteria for the following cybersecurity requirement.

Control Metadata:
- control_id: {control_id}
- control: {control}
- subcontrol: {subcontrol}
- framework: {framework}

Use the following framework-specific guidance where relevant:
{framework_hint}


========================================================
TEST OF DESIGN (TOD) – WHAT MUST EXIST
========================================================
Generate exactly {num_tod} TOD criteria.

Each TOD criterion MUST:
- Be a governance-level requirement describing what must exist in the control’s design
  (policies, procedures, definitions, roles, responsibilities, documentation, scope, or frequency).
- Be written as ONE logical criterion, but may contain multiple sentences when needed to express a complete requirement clearly.
- Use formal compliance language such as “Existence of…”, “Clear definition of…”, “Documented procedures for…”, “Assignment of responsibilities for…”.
- Not include operational or audit verbs such as verify, validate, extract, compare, inspect, review, check, sample, reconcile, correlate.
- Stay strictly within the scope and intent of the subcontrol; do not introduce unrelated requirements.
- Be measurable, specific, unambiguous, and independent of other criteria.

Collectively, TOD criteria must (where applicable):
1) Establish the presence of required policy, procedure, or documentation.
2) Define the scope, categorization, or coverage of the control.
3) Clarify roles and responsibilities.
4) Define update, maintenance, review, or lifecycle expectations.

TOD criteria should follow professional cybersecurity audit design style but not imitate any specific individual auditor.


========================================================
TEST OF OPERATING EFFECTIVENESS (TOE) – HOW TO TEST
========================================================
Generate exactly {num_toe} TOE criteria.

Each TOE criterion MUST:
- Be a concrete audit procedure describing how an auditor verifies the control.
- Begin with an action verb such as: “Extract”, “Review”, “Compare”, “Validate”, “Match”, “Cross-check”, “Inspect”, “Sample”, “Identify”.
- Be ONE logical criterion, but may contain multiple sentences if needed (for example: evidence step + validation condition + exception handling).
- Specify:
  • What evidence is used (logs, inventories, HR records, system exports, reports, timestamps, approvals, configurations).  
  • Which fields or attributes are important (owner, timestamp, identifier, type, status).  
  • What validation or comparison the auditor performs (completeness, accuracy, consistency, timeliness, correlation).  
- Reflect authentic audit workflow: evidence gathering, extraction, verification, cross-system comparison, and exception identification.
- Not include governance wording such as “The entity shall…” or “The policy must define…”, which belong to TOD.

Collectively, TOE criteria must include:
1) Evidence extraction from authoritative systems.
2) Field-level validation and completeness checks.
3) Date/timestamp review when relevant.
4) Cross-system reconciliation where applicable.
5) Identification of missing, outdated, or inconsistent records.

TOE criteria must read like standard audit procedures that any qualified auditor could perform.


========================================================
STRICT OUTPUT FORMAT — RETURN ONLY THIS JSON
========================================================
Return ONLY the JSON object below, with no text before or after:

{{
  "control_id": "{control_id}",
  "control": "{control}",
  "subcontrol": "{subcontrol}",
  "framework": "{framework}",
  "llm_content": {{
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
}}

Rules:
- No markdown, no backticks, no commentary.
- Only valid JSON, directly parseable by json.loads() in Python without modification.
- Each "criteria" value must be a human-readable natural-language string.
- Multi-sentence criteria are allowed, but must remain ONE coherent criterion.
- Do not hallucinate unrelated requirements; stay strictly aligned to the subcontrol.

Return only the JSON object.
""".strip()


# ---------------------------------------------------------------------
# Helper: Build messages list for chat completion APIs
# ---------------------------------------------------------------------

def build_tod_toe_messages(
    control_id: str,
    control: str,
    subcontrol: str,
    framework: str,
    num_tod: int,
    num_toe: int,
) -> List[Dict[str, str]]:
    """
    Build a list of messages suitable for chat-based LLM APIs
    (OpenAI, Azure OpenAI, etc.).

    Example use:

        messages = build_tod_toe_messages(
            control_id="T1.2.1.2",
            control="T1.2.1.2: Inventory of Information and other Associated Assets",
            subcontrol="11.2.1.2: The entity shall identify and maintain an up-to-date inventory of information assets within the entity.",
            framework="UAE Information Assurance Regulation",
            num_tod=3,
            num_toe=3,
        )

        response = client.chat.completions.create(
            model="phi-4",
            messages=messages,
            temperature=0.0,
            max_tokens=700,
        )

    """
    system_prompt = get_tod_toe_system_prompt(framework)
    user_prompt = get_tod_toe_user_prompt(
        control_id=control_id,
        control=control,
        subcontrol=subcontrol,
        framework=framework,
        num_tod=num_tod,
        num_toe=num_toe,
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
