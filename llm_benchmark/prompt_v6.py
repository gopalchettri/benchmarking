# ============================================================================
# OPTIMIZED TOD/TOE PROMPT SYSTEM - PRODUCTION READY
# ============================================================================

FRAMEWORK_HINT_MAP = {
    "UAE Information Assurance Regulation": "Use formal regulatory language: 'the entity shall', 'the organization must'.",
    "UAE IA": "Use formal regulatory language: 'the entity shall', 'the organization must'.",
    "DESC ISR": "Use concise, security-operations language aligned with Dubai ISR technical controls.",
    "NCA ECC": "Use formal language for critical infrastructure protection contexts.",
    "SWIFT CSCF": "Focus on payment messaging infrastructure, secure zones, and SWIFT-specific terminology.",
    "PCI DSS": "Emphasize cardholder data, payment systems, and PCI technical terminology.",
    "ISO 27001": "Use ISMS terminology: information assets, risk treatment, controls.",
    "_default_": "Use professional cybersecurity and risk management terminology."
}

IMPLEMENTATION_GUIDANCE_MAP = {
    "T1.2.2": """The entity should identify assets relevant in the lifecycle of information and document their importance. The lifecycle of information should include creation, processing, storage, transmission, deletion, destruction. Information assets inventory should include but not limited to: Hardware (Server, Network devices, Storage devices, Workstations), People (Chief Technology/Information Director), Database servers, Applications, Financial data, Customer data, Facilities, Data centers. The asset owner should be responsible for ensuring that information and assets associated with information systems are appropriately identified, classified and maintained.""",
    "T1.3.1": """The entity should identify assets relevant in the lifecycle of information and document their importance. Classification should account for value, legal requirements, sensitivity, and criticality. Ownership may be allocated based on business process. The entity shall define and implement an information classification scheme. Subcontrol text and control text together describe the requirement and are the main sources of truth.""",
    "T1.3.2": """The information classification scheme should include procedures describing how to apply classification labels to information assets based on their value, legal requirements, sensitivity, and criticality. The procedures should include guidance for asset owners on how to classify information and document their classifications.""",
    "T2.1.1": """Access control policies should define who can access what information and under what circumstances. Policies should address physical and logical access controls, including authentication requirements, authorization rules, and access review procedures.""",
    "T2.2.1": """User access management procedures should include user registration, provisioning, de-provisioning, privilege management, review of user access rights, and removal or adjustment of access rights. The entity must ensure that access is granted based on business need and removed when no longer required.""",
    "T2.3.1": """The entity should implement strong authentication mechanisms including multi-factor authentication where appropriate. Password policies should define complexity requirements, expiration periods, and restrictions on password reuse.""",
    "ISR-2.1": """Access control mechanisms should be implemented at network, system, and application layers. The organization should maintain access control lists, review access permissions regularly, and ensure least privilege principle is enforced.""",
    "ISR-2.3": """Multi-factor authentication must be implemented for privileged access to critical systems. MFA should use at least two different authentication factors: something you know (password), something you have (token, smart card), or something you are (biometric).""",
    "ISR-4.2": """Encryption for data at rest should use industry-standard algorithms (AES-256 or equivalent). Key management procedures must be documented including key generation, storage, rotation, and destruction. Encryption should be applied to databases, file systems, and backup media containing sensitive information.""",
    "ISR-5.1": """Security logs should be generated for authentication events, access to sensitive data, administrative actions, and security-relevant system events. Logs must be retained for minimum periods as defined by regulatory requirements and protected from unauthorized modification.""",
    "ISR-5.2": """Log monitoring procedures should include real-time alerting for critical security events, regular review of logs for anomalies, and incident response procedures when suspicious activities are detected.""",
    "T3.4.1": """Cryptographic controls should be implemented based on risk assessment and data classification. The entity should maintain a cryptographic key management policy covering key generation, distribution, storage, rotation, backup, and destruction.""",
    "T4.3.1": """Security monitoring should include continuous monitoring of security events, log analysis, intrusion detection, and regular security assessments. The entity should define what constitutes normal vs. suspicious activity and establish alerting thresholds.""",
    "T4.3.2": """Incident response procedures should define roles and responsibilities, incident classification criteria, escalation procedures, communication protocols, and evidence collection requirements. The entity should conduct regular incident response drills.""",
    "T5.2.1": """Backup procedures should define backup frequency, retention periods, backup media security, and backup testing procedures. The entity should maintain backup copies in geographically separate locations and test restoration procedures regularly.""",
    "_default_": """Implementation guidance should be derived from the control and subcontrol text. Focus on how the control is typically implemented in practice and what evidence would demonstrate effective implementation."""
}

def get_implementation_guidance(control_id: str) -> str:
    return IMPLEMENTATION_GUIDANCE_MAP.get(control_id, IMPLEMENTATION_GUIDANCE_MAP["_default_"])

def get_system_prompt(framework: str) -> str:
    framework_hint = FRAMEWORK_HINT_MAP.get(framework, FRAMEWORK_HINT_MAP["_default_"])
    return f"""You are an expert cybersecurity compliance auditor specializing in {framework}.

Your task: Generate audit criteria that faithfully express the control/subcontrol intent using their original terminology. Do not introduce requirements beyond what is stated or clearly implied.

Framework style: {framework_hint}

Output: Return valid JSON starting with '{{' and ending with '}}'. Use double quotes, no markdown, no code blocks. Must be parseable by json.loads()."""

def get_combined_tod_toe_prompt(
    control_id: str,
    control: str,
    subcontrol: str,
    framework: str,
    num_tod: int,
    num_toe: int,
    implementation_guidance: str = None
) -> str:
    framework_hint = FRAMEWORK_HINT_MAP.get(framework, FRAMEWORK_HINT_MAP["_default_"])
    if implementation_guidance is None:
        implementation_guidance = get_implementation_guidance(control_id)
    guidance_section = f"\nImplementation Guidance: {implementation_guidance}" if implementation_guidance else ""
    
    return f"""Generate {num_tod} Test of Design (TOD) and {num_toe} Test of Operating Effectiveness (TOE) criteria for this cybersecurity control.

## INPUT
Framework: {framework}
Control ID: {control_id}
Control: {control}
Subcontrol: {subcontrol}{guidance_section}

Framework style: {framework_hint}

## REQUIREMENTS

**TOD Criteria** - Verify control design exists and is documented:
- Test exactly ONE design element per criterion
- Focus on policies, procedures, defined roles, documented processes
- Verifiable through document review or interviews
- Coverage: Governance/policy → Roles/responsibilities → Documented procedures

**TOE Criteria** - Verify control operates effectively in practice:
- Test exactly ONE operational aspect per criterion
- Focus on implementation evidence: logs, records, configurations, system checks
- Use action verbs: verify, validate, inspect, examine, test, review, confirm
- Coverage: Implementation evidence → Operational records → Monitoring/review evidence

## EXAMPLES

**Example 1 - UAE IAR (Information Classification)**

Control: "The entity shall define and implement an information classification scheme based on information value, legal requirements, sensitivity, and criticality"

TOD Criteria:
[
  {{"id": 1, "criteria": "Verify that a documented policy exists for defining the information classification scheme"}},
  {{"id": 2, "criteria": "Confirm that there is an approved information classification framework that addresses value, legal requirements, sensitivity, and criticality of information"}},
  {{"id": 3, "criteria": "Validate that procedures are in place to ensure all relevant types of information are classified according to the established scheme"}},
  {{"id": 4, "criteria": "Verify that roles and responsibilities related to implementation and maintenance of the information classification scheme are clearly defined"}}
]

TOE Criteria:
[
  {{"id": 1, "criteria": "Extract the information classification policy and verify that each record contains a classification field that corresponds to one of the classification levels defined in the policy"}},
  {{"id": 2, "criteria": "Select a sample of information assets (files, databases, system entries) and verify that each has a valid classification value assigned"}},
  {{"id": 3, "criteria": "Review access control configurations and confirm that permissions align with the assigned classification levels"}},
  {{"id": 4, "criteria": "Examine classification assignment records and validate that information assets are reviewed and reclassified when their value or sensitivity changes"}}
]

**Example 2 - DESC ISR (Access Control)**

Control: "The organization shall implement multi-factor authentication for privileged access to critical systems"

TOD Criteria:
[
  {{"id": 1, "criteria": "Verify that a documented policy exists requiring MFA for all privileged accounts accessing critical systems"}},
  {{"id": 2, "criteria": "Confirm that critical systems are identified and documented with their privileged access requirements"}},
  {{"id": 3, "criteria": "Validate that procedures define the MFA implementation process for privileged users"}}
]

TOE Criteria:
[
  {{"id": 1, "criteria": "Inspect MFA configuration settings on critical systems and confirm that MFA is enabled for all privileged accounts"}},
  {{"id": 2, "criteria": "Select a sample of privileged user accounts and verify that MFA is enforced during authentication"}},
  {{"id": 3, "criteria": "Review authentication logs and validate that privileged access attempts require and complete MFA verification"}}
]

**Example 3 - UAE IAR (Asset Inventory)**

Control: "The entity shall maintain an up-to-date inventory of information assets within the entity"

TOD Criteria:
[
  {{"id": 1, "criteria": "Verify existence of documented asset inventory procedures"}},
  {{"id": 2, "criteria": "Confirm that roles responsible for maintaining the information asset inventory are defined"}}
]

TOE Criteria:
[
  {{"id": 1, "criteria": "Obtain the current asset inventory database and verify it contains information assets with identification details"}},
  {{"id": 2, "criteria": "Select a sample of information assets and confirm they are recorded in the inventory with accurate classification"}}
]

## OUTPUT FORMAT

Return only this JSON structure:

{{
  "control_id": "{control_id}",
  "control": "{control}",
  "subcontrol": "{subcontrol}",
  "framework": "{framework}",
  "llm_content": {{
    "llm_tod_criterias": [
      {{"id": 1, "criteria": "First TOD criteria in natural language"}},
      {{"id": 2, "criteria": "Second TOD criteria in natural language"}}
    ],
    "llm_toe_criterias": [
      {{"id": 1, "criteria": "First TOE criteria in natural language"}},
      {{"id": 2, "criteria": "Second TOE criteria in natural language"}}
    ]
  }}
}}

Generate exactly {num_tod} TOD criteria and {num_toe} TOE criteria. Return only the JSON object."""
