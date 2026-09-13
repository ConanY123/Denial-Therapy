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
def route_event(node_input: DenialEvent) -> Event:
    """Classifies the event based on the supported denial reason codes."""
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
        
    if re.search(r'\bID[:\s]*[A-Z0-9]{5,}\b', desc, re.I) or re.search(r'\bID[:\s]*[A-Z0-9]{5,}\b', patient, re.I):
        redacted.add("Member ID")
        desc = re.sub(r'\bID[:\s]*[A-Z0-9]{5,}\b', '[REDACTED MEMBER ID]', desc, flags=re.I)
        patient = re.sub(r'\bID[:\s]*[A-Z0-9]{5,}\b', '[REDACTED MEMBER ID]', patient, flags=re.I)
        
    if re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', desc) or re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', patient):
        redacted.add("DOB")
        desc = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[REDACTED DOB]', desc)
        patient = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '[REDACTED DOB]', patient)
        
    if re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', patient):
        redacted.add("Name")
        patient = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[REDACTED NAME]', patient)

    node_input.denial_description = desc
    node_input.patient_info = patient
    
    # 2. Prompt Injection Defense
    injection_patterns = [r'ignore previous', r'force approval', r'override', r'bypass', r'fabricate', r'always approve']
    is_injection = any(re.search(p, desc, re.I) for p in injection_patterns)
    
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
    input_schema=DenialEvent,
    description="A workflow to process and appeal medical claim denials."
)
