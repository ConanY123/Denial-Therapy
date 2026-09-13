from typing import Any
import re
from pydantic import BaseModel, Field
from google.adk.workflow import Workflow, node
from google.adk.agents import LlmAgent
from google.adk.events import Event, RequestInput
from google.adk.agents.context import Context
from google.genai import types

from .config import SUPPORTED_REASON_CODES, PAYER_NAME, LLM_MODEL

# 1. Schemas from fastapis pydantic
class DenialEvent(BaseModel):
    claim_id: str
    payer: str
    cpt_code: str
    denial_reason_code: str
    billed_amount: float
    denial_description: str
    patient_info: str

class DraftOutput(BaseModel):
    appeal_letter_text: str = Field(description="The full drafted body of the appeal letter.")
    cited_cpt_code: str = Field(description="The specific CPT code referenced in the letter.")
    cited_denial_reason: str = Field(description="The specific denial reason addressed.")
    payer_name: str = Field(description="The name of the payer the letter is addressed to.")

# 2. Router Node
@node
def route_event(node_input: Any) -> Event:
    """Classifies the event based on the supported denial reason codes."""

    if isinstance(node_input, DenialEvent):
        node_input = node_input
    elif isinstance(node_input, types.Content):
        text = next(
            (p.text for p in (node_input.parts or []) if getattr(p, "text", None)),
            None,
        )
        if text is None:
            raise ValueError("No text part found in Content input for DenialEvent.")
        node_input = DenialEvent.model_validate_json(text)
    elif isinstance(node_input, str):
        node_input = DenialEvent.model_validate_json(node_input)
    elif isinstance(node_input, dict):
        node_input = DenialEvent.model_validate(node_input)
    else:
        raise TypeError(
            f"Unsupported input type for route_event: {type(node_input).__name__}"
        )

    normalized_code = node_input.denial_reason_code.lower().strip()
    if normalized_code in SUPPORTED_REASON_CODES:
        return Event(
            output=node_input, 
            state={"original_event": node_input.model_dump(), "classification": "supported"}
        )
    else:
        return Event(
            output=node_input, 
            state={"original_event": node_input.model_dump(), "classification": "unsupported"}
        )

# 2.5 Security node
@node
def security_checkpoint(ctx: Context, node_input: DenialEvent) -> Event:
    """Scrubs PII and blocks prompt injection before reaching the LLM."""
    desc = node_input.denial_description
    patient = node_input.patient_info
    redacted = set()
    
    # 1. Scrubbing info(PII)
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', desc) or re.search(r'\b\d{3}-\d{2}-\d{4}\b', patient):
        redacted.add("SSN")
        desc = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED SSN]', desc)
        patient = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED SSN]', patient)
        
    # Member ID: fail-safe scrubbing that over-redacts rather than risk leaking
    # PHI. Catches labeled IDs ("Member ID: X", "ID X", "Member # X") AND bare
    # identifier tokens (a letter + 6+ digits, or a run of 8+ digits) so an
    # unlabeled "M123456789" is still scrubbed. SSNs are already redacted above.
    #
    # Known-safe exception: claim/CPT numbers are NOT PII, so we never redact
    # the event's own claim_id or cpt_code, nor CLM-style claim references. We
    # only exempt these specific known tokens; every other bare long token is
    # still treated as a potential member ID (the safe default).
    claim_safelist = {
        node_input.claim_id.upper(),
        node_input.cpt_code.upper(),
    }
    # Also treat any CLM-<digits> style token as a claim reference, not PII.
    claim_ref_re = re.compile(r'\bCLM-\w+\b', re.I)
    for m in claim_ref_re.finditer(f"{desc} {patient}"):
        claim_safelist.add(m.group(0).upper())

    def _redact_member_id(match: "re.Match[str]") -> str:
        token = match.group(0)
        if token.upper() in claim_safelist:
            return token  # known claim/CPT reference — leave untouched
        redacted.add("Member ID")
        return "[REDACTED MEMBER ID]"

    # Labeled form: an explicit "Member ID: X" label is unambiguous PII.
    labeled_id_re = re.compile(r'((?:member\s*(?:id|#|number)|id)[:#\s]+)([A-Z0-9]{5,})', re.I)
    if labeled_id_re.search(desc) or labeled_id_re.search(patient):
        redacted.add("Member ID")
        desc = labeled_id_re.sub(r'\1[REDACTED MEMBER ID]', desc)
        patient = labeled_id_re.sub(r'\1[REDACTED MEMBER ID]', patient)

    # Bare tokens: letter + 6+ digits, or a run of 8+ digits, excluding
    # safelisted claim/CPT references via the replacement function.
    for bare_re in (re.compile(r'\b[A-Z]\d{6,}\b', re.I), re.compile(r'\b\d{8,}\b')):
        desc = bare_re.sub(_redact_member_id, desc)
        patient = bare_re.sub(_redact_member_id, patient)
        
    # Only redact a date when it is explicitly labeled as a birth date.
    # A generic date regex would over-redact claim/service dates (e.g. a
    # processing date) and mislabel them as DOB. We require a birth-date cue
    # (DOB, D.O.B., date of birth, born, birthdate) immediately preceding the
    # date, and redact only the date itself (keeping the cue for readability).
    dob_pattern = re.compile(
        r'((?:D\.O\.B\.|\b(?:DOB|date of birth|birth\s*date|born)\b)[:\s]*)'
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        re.I,
    )
    if dob_pattern.search(desc) or dob_pattern.search(patient):
        redacted.add("DOB")
        desc = dob_pattern.sub(r'\1[REDACTED DOB]', desc)
        patient = dob_pattern.sub(r'\1[REDACTED DOB]', patient)
        
    # Name: the old rule redacted ANY two capitalized words, which clobbers
    # org names ("Medical Center", "Blue Cross"). Instead redact a name only
    # when it is either (a) preceded by a name cue, or (b) the leading token(s)
    # of patient_info (the conventional position for the patient's name).
    name_token = r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+'  # two+ capitalized words
    labeled_name = re.compile(
        rf'((?:patient(?:\s*name)?|name|insured|beneficiary|member)\b[:\s]+)({name_token})',
        re.I,
    )
    leading_name = re.compile(rf'^\s*({name_token})')

    name_found = False
    if labeled_name.search(patient):
        name_found = True
        patient = labeled_name.sub(r'\1[REDACTED NAME]', patient)
    if leading_name.search(patient):
        name_found = True
        patient = leading_name.sub('[REDACTED NAME]', patient)
    if labeled_name.search(desc):
        name_found = True
        desc = labeled_name.sub(r'\1[REDACTED NAME]', desc)
    if name_found:
        redacted.add("Name")

    node_input.denial_description = desc
    node_input.patient_info = patient
    
    # 2. Prompt Injection Defense
    # Fail-safe: match the common attack vocabulary rather than a few fixed
    # phrases. Patterns are intentionally flexible (e.g. "ignore <any>
    # instructions" covers prior/previous/above/all) and we scan BOTH the
    # denial_description AND patient_info, since an attacker can hide an
    # instruction in either free-text field. Any match diverts to human_review.
    injection_patterns = [
        r'ignore\s+(?:all\s+|the\s+|any\s+)?(?:prior|previous|above|preceding|earlier)?\s*instructions',
        r'disregard\s+(?:all\s+|the\s+|any\s+)?(?:prior|previous|above|preceding|earlier)?\s*(?:instructions|context)',
        r'develop(?:er)?\s*mode',
        r'jail\s*break',
        r'(?:reveal|expose|show|print|repeat|leak|output)\s+(?:your|the)?\s*(?:system\s*)?prompt',
        r'system\s*prompt',
        r'you\s+are\s+now',
        r'act\s+as\s+(?:a\s+)?(?:different|another)',
        r'force\s+approval',
        r'approve\s+(?:automatically|this\s+claim\s+automatically|it\s+automatically)',
        r'auto[-\s]*approve',
        r'always\s+approve',
        r'override',
        r'bypass',
        r'fabricate',
    ]
    injection_scan_text = f"{desc}\n{patient}"
    is_injection = any(
        re.search(p, injection_scan_text, re.I) for p in injection_patterns
    )
    
    if is_injection:
        return Event(output=node_input, route="human_review", state={"redacted_categories": list(redacted), "security_flag": "prompt_injection"})
        
    classification = ctx.state.get("classification", "supported")
    if classification == "supported":
        return Event(output=node_input, route="clean_supported", state={"redacted_categories": list(redacted)})
    else:
        return Event(output=node_input, route="human_review", state={"redacted_categories": list(redacted)})

# 3. LLM Agent Node
draft_appeal = LlmAgent(
    name="draft_appeal",
    model=LLM_MODEL,
    instruction=(
        f"You are a medical billing expert drafting appeal letters for {PAYER_NAME}. "
        "Based on the provided denial event data, draft a structured and professional "
        "appeal letter. You must specifically cite the CPT code and the exact denial "
        "reason code in your rationale."
    ),
    output_schema=DraftOutput,
    output_key="draft",
)

# 4. Human in the loop(HITL) Node
@node(rerun_on_resume=True)
async def human_review(ctx: Context, node_input: Any):
    """Pauses execution for a human reviewer to approve, edit, or reject."""
    if not ctx.resume_inputs:
        msg = "Unknown input type received."
        if isinstance(node_input, DenialEvent):
            if ctx.state.get("security_flag") == "prompt_injection":
                msg = f"SECURITY ALERT: Prompt injection blocked. PII scrubbed: {ctx.state.get('redacted_categories')}"
            else:
                msg = "This code is unsupported, no draft was generated."
        elif isinstance(node_input, dict):
            try:
                # ADK automatically converts output_schema models to dict
                draft = DraftOutput.model_validate(node_input)
                msg = f"Here's a drafted appeal, approve/edit/reject:\n\nDraftOutput details: {node_input}"
            except Exception:
                msg = "Unknown dict format received."
                
        yield Event(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=msg)]
            )
        )
        yield RequestInput(interrupt_id="review", message=msg)
        return
        
    yield Event(output={"action": ctx.resume_inputs["review"]})

# 5. Wiring the Graph Workflow
edges = [
    ('START', route_event),
    (route_event, security_checkpoint),
    (security_checkpoint, {
        "clean_supported": draft_appeal, 
        "human_review": human_review
    }),
    (draft_appeal, human_review)
]

root_agent = Workflow(
    name="denial_appeal_workflow",
    edges=edges,
    # NOTE: We deliberately do NOT set input_schema=DenialEvent here.
    # When input_schema is set on the Workflow, ADK's BaseNode._validate_input_data
    # runs BEFORE route_event and, on any validation failure, falls through to
    # validating the raw Content object against DenialEvent — producing a
    # confusing "input_type=Content" ValidationError that masks the real issue
    # (e.g. a malformed field or plain-text input). route_event already handles
    # Content / str / dict / DenialEvent inputs and parses them itself, so we let
    # it own input coercion and surface clearer errors.
    description="A workflow to process and appeal medical claim denials."
)
