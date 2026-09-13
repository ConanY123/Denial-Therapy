import os
import base64
import json
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# ADK imports
from google.adk.apps import App
from google.adk.runners import InMemoryRunner
from google.genai import types

# Import the root_agent and DenialEvent schema
from appealerAgent.agent import root_agent, DenialEvent

# 1. Telemetry Configuration
# By default, ADK only exports OTEL data to the cloud when using the 'adk web' CLI
# (which explicitly sets up Google Cloud exporters). Since we are instantiating 
# the App and InMemoryRunner directly, OTEL cloud exports are natively disabled.

# 2. Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Denial Therapy Ambient Agent")

# Initialize ADK App and Runner
adk_app = App(name="ambient_app", root_agent=root_agent)
# Note: Paused human_review sessions are lost on service restart, and resuming a 
# paused review via this Pub/Sub endpoint isn't implemented yet. New events are 
# ingested, but there's no resume path through this endpoint. 
runner = InMemoryRunner(app=adk_app)

class PubSubMessage(BaseModel):
    data: str
    messageId: str

class PubSubRequest(BaseModel):
    message: PubSubMessage
    subscription: str

class ResumeRequest(BaseModel):
    session_id: str
    user_id: str
    decision: str
    edited_text: str | None = None

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "Denial Therapy Ambient Agent"}

@app.post("/pubsub")
async def handle_pubsub(payload: PubSubRequest):
    """
    Accepts Pub/Sub trigger messages and feeds them into the root_agent workflow.
    """
    try:
        # Normalize the subscription path down to a short name
        # e.g., 'projects/my-project/subscriptions/my-sub' -> 'my-sub'
        sub_name = payload.subscription.split("/")[-1]
        
        # Create a readable session record name
        session_user_id = f"{sub_name}-{payload.message.messageId}"
        
        logger.info(f"Processing message {payload.message.messageId} from {sub_name}")
        
        # Decode base64 data
        decoded_bytes = base64.b64decode(payload.message.data)
        decoded_str = decoded_bytes.decode('utf-8')
        
        # Parse into DenialEvent dict to ensure it's valid JSON
        # It's expected to be JSON that matches DenialEvent schema
        event_data = json.loads(decoded_str)
        
        # Validate data against schema
        denial_event = DenialEvent(**event_data)
        
    except Exception as e:
        logger.error(f"Failed to parse Pub/Sub message: {e}")
        raise HTTPException(status_code=400, detail="Invalid message format")

    try:
        # Create a session for this specific event
        session = await runner.session_service.create_session(
            app_name="ambient_app", 
            user_id=session_user_id
        )
        
        logger.info(f"Created session {session.id} for user {session_user_id}")
        
        # Feed the event into the workflow
        async for event in runner.run_async(
            user_id=session_user_id,
            session_id=session.id,
            new_message=types.Content(
                role="user", 
                parts=[types.Part.from_text(text=denial_event.model_dump_json())]
            )
        ):
            if event.node_info and event.output is not None:
                logger.info(f"[{event.node_info.path} OUTPUT] -> {event.output}")
                if "draft_appeal" in event.node_info.path:
                    logger.info(f"DraftOutput details: {event.output}")
                
            if getattr(event, "long_running_tool_ids", None):
                fc = event.content.parts[0].function_call
                if fc and fc.name == "adk_request_input":
                    logger.info(f"[PAUSED] Workflow paused awaiting human review. id: {fc.id}, Message: {fc.args.get('message')}")
                    if fc.args.get('draft'):
                        logger.info(f"DraftOutput details: {fc.args.get('draft')}")
                
        logger.info(f"Workflow execution completed/paused for message {payload.message.messageId}")
        return {"status": "success", "session_id": session.id}
        
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail="Workflow execution failed")

@app.post("/resume")
async def resume_workflow(req: ResumeRequest):
    """
    Resumes a paused workflow session with a human reviewer's decision.
    """
    try:
        # 1. Look up the existing paused session
        session = await runner.session_service.get_session(
            app_name="ambient_app",
            user_id=req.user_id,
            session_id=req.session_id
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Check if session was already resolved
        for event in session.events:
            if event.output and isinstance(event.output, dict) and "action" in event.output:
                action_data = event.output.get("action", {})
                decision = action_data.get("decision") if isinstance(action_data, dict) else action_data
                raise HTTPException(
                    status_code=409,
                    detail=f"This session was already resolved with decision: {decision}"
                )
            
        logger.info(f"Resuming session {req.session_id} for user {req.user_id} with decision: {req.decision}")
        
        # 2. Feed the decision back into the workflow
        response_payload = {"decision": req.decision}
        if req.decision == "edit" and req.edited_text:
            response_payload["edited_text"] = req.edited_text
            
        # Create the resume payload. The id matches the adk_request_input id ("review")
        resume_message = types.Content(
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="adk_request_input",
                        id="review",
                        response=response_payload
                    )
                )
            ]
        )
        
        # Run the workflow with the resume message
        final_output = None
        async for event in runner.run_async(
            user_id=req.user_id,
            session_id=req.session_id,
            new_message=resume_message
        ):
            logger.info(f"[RESUME EVENT] -> {event}")
            if event.node_info and event.output is not None:
                logger.info(f"[{event.node_info.path} OUTPUT] -> {event.output}")
                final_output = event.output
                
        # 3. Return the final workflow state/output
        return {"status": "success", "session_id": req.session_id, "final_output": final_output}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow resume failed: {e}")
        raise HTTPException(status_code=500, detail="Workflow resume failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
