import json
import os
import re
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

# Best Practice: Keep API keys in your .env file: GROQ_API_KEY=your_key_here
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("❌ GROQ_API_KEY not found. Check your .env file.")

# ----------------------------
# Mock Tool Function
# ----------------------------
def mock_lead_capture(name, email, platform):
    print(f"\n✅ Lead captured successfully: {name}, {email}, {platform}\n")
    return "Lead captured successfully."

# ----------------------------
# State Definition
# ----------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str
    lead_data: dict

# ----------------------------
# Load Knowledge Base
# ----------------------------
try:
    with open("data.json", "r") as f:
        knowledge_base = json.load(f)
    # Convert KB to a formatted string once to save processing
    kb_string = json.dumps(knowledge_base, indent=2)
except FileNotFoundError:
    print("⚠️ data.json not found. Using empty knowledge base.")
    kb_string = "{}"

# ----------------------------
# Initialize LLM
# ----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0 # Set temperature to 0 for more deterministic classification
)

# ----------------------------
# Helper Functions
# ----------------------------
def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else None

# ----------------------------
# 1. Intent Classifier
# ----------------------------
def intent_classifier(state: AgentState):
    user_message = state["messages"][-1].content.lower()
    lead_data = state.get("lead_data", {})
    previous_intent = state.get("intent", "")

    valid_lead_data = {k: v for k, v in lead_data.items() if v}

    # 1. Escape Hatch: Allow the user to cancel out of the form
    if user_message in ["cancel", "stop", "nevermind", "quit", "exit", "abort"]:
        print("[DEBUG] User cancelled flow.")
        # Reset the intent and clear the lead_data state
        return {"intent": "casual_greeting", "lead_data": {}}

    # 2. STATE LOCK: If we started collecting leads, stay here until finished!
    if previous_intent == "high_intent" and len(valid_lead_data) < 3:
        # Exception: Allow user to break out if they ask a new product question
        if any(word in user_message for word in ["?", "price", "cost", "refund", "plans", "discount"]):
            pass # Release the lock and let the LLM re-classify
        else:
            print("[DEBUG] State Lock Active: Routing directly to lead capture.")
            return {"intent": "high_intent"}

    # 3. Rule-based (fast)
    if user_message in ["hi", "hello", "hey", "greetings"]:
        return {"intent": "casual_greeting"}

    if any(x in user_message for x in ["buy", "subscribe", "start", "signup", "trial"]):
        return {"intent": "high_intent"}

    # 4. Fallback LLM Classification
    prompt = f"""
    Analyze the following message and classify it into EXACTLY ONE of these categories:
    1. casual_greeting (greetings, thank yous, goodbyes, conversational filler)
    2. product_inquiry (questions about pricing, features, plans, refunds)
    3. high_intent (explicit statements of wanting to buy, upgrade, or sign up)

    Message: "{user_message}"
    Respond with ONLY the exact category name. Do not include any other text.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    intent = response.content.strip().lower()

    # Stricter parsing
    if "high_intent" in intent:
        final_intent = "high_intent"
    elif "product_inquiry" in intent:
        final_intent = "product_inquiry"
    else:
        final_intent = "casual_greeting"

    print(f"[DEBUG] Intent classified as: {final_intent}")
    return {"intent": final_intent}

# ----------------------------
# 2. Greeting Handler
# ----------------------------
def handle_greeting(state: AgentState):
    response = llm.invoke(
        [SystemMessage(content="You are a polite assistant for AutoStream. Reply briefly and ask how you can help.")]
        + state["messages"]
    )
    return {"messages": [response]}

# ----------------------------
# 3. RAG (Simple Retrieval)
# ----------------------------
def retrieve_and_answer(state: AgentState):
    # Pass the entire KB string instead of brittle keyword matching
    system_prompt = f"""
    You are an AutoStream assistant.
    Answer the user's question using ONLY the following knowledge base:
    
    {kb_string}

    If the answer is not in the data, politely say "I don't know."
    """

    response = llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    return {"messages": [response]}

# ----------------------------
# 4. Lead Capture
# ----------------------------
def capture_lead(state: AgentState):
    lead_data = state.get("lead_data", {})
    user_message = state["messages"][-1].content
    user_message_lower = user_message.lower()

    # Extract Email (Regex is foolproof for this, bypasses LLM entirely)
    email = extract_email(user_message)
    if email:
        lead_data["email"] = email

    # Bulletproof extraction prompt with explicit schema
    extraction_prompt = f"""
    Extract the user's Name and Creator Platform from the message.
    
    CRITICAL RULES:
    1. Output ONLY a valid JSON object. No markdown, no extra text.
    2. You MUST use exactly these two lowercase keys: "name" and "platform".
    3. If a value is missing or the user doesn't give a real human name, use null.
    
    Example Output:
    {{"name": "John Doe", "platform": "YouTube"}}
    
    Message: "{user_message}"
    """
    
    try:
        # Strict System Persona
        ext_response = llm.invoke([
            SystemMessage(content="You are a strict JSON bot. Output only valid JSON with keys 'name' and 'platform'."),
            HumanMessage(content=extraction_prompt)
        ])
        
        # Aggressive cleaning
        clean_json = ext_response.content.replace('```json', '').replace('```', '').strip()
        start_idx = clean_json.find('{')
        end_idx = clean_json.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            clean_json = clean_json[start_idx:end_idx]
            
        extracted_raw = json.loads(clean_json)
        
        # Normalize all keys to lowercase to prevent capitalization bugs
        extracted = {k.lower(): v for k, v in extracted_raw.items()}
        
        name_val = extracted.get("name")
        platform_val = extracted.get("platform")

        # Save valid name
        if name_val and isinstance(name_val, str) and len(name_val.split()) < 4 and name_val.lower() != "null":
            if not lead_data.get("name"):
                lead_data["name"] = name_val
                
        # Save valid platform
        if platform_val and isinstance(platform_val, str) and platform_val.lower() != "null":
            if not lead_data.get("platform"):
                lead_data["platform"] = platform_val
                
    except Exception as e:
        print(f"[DEBUG] JSON Extraction failed: {e}")
        pass

    # Safety Net: Rule-based fallback for Platforms
    # If the LLM completely fails, this ensures we still catch common platforms
    if not lead_data.get("platform"):
        for p in ["youtube", "instagram", "tiktok", "facebook", "twitch", "twitter", "x", "linkedin"]:
            if p in user_message_lower:
                lead_data["platform"] = p.title()
                break

    # Clean up lead_data to remove any accidental empty values
    lead_data = {k: v for k, v in lead_data.items() if v}

    # Check missing fields
    required = ["name", "email", "platform"]
    missing = [k for k in required if not lead_data.get(k)]

    if not missing:
        mock_lead_capture(
            lead_data.get("name"),
            lead_data.get("email"),
            lead_data.get("platform")
        )
        return {
            "intent": "casual_greeting",
            "lead_data": {}, # Wipe the memory clean after successful capture!
            "messages": [AIMessage(content="✅ Thanks! We've captured your details. Our team will reach out soon.")]
        }

    # Ask for missing details conversationally (Fixed the hardcoded "Pro plan" text)
    missing_str = ", ".join(missing).title()
    return {
        "lead_data": lead_data,
        "messages": [AIMessage(content=f"To get you set up with your subscription, I just need a bit more info. Could you please provide your: {missing_str}?")]
    }

# ----------------------------
# 5. Routing Logic
# ----------------------------
def route_intent(state: AgentState):
    if state["intent"] == "casual_greeting":
        return "greeting"
    elif state["intent"] == "high_intent":
        return "lead_capture"
    else:
        return "rag_qa"

# ----------------------------
# 6. Build Graph
# ----------------------------
workflow = StateGraph(AgentState)

workflow.add_node("classify", intent_classifier)
workflow.add_node("greeting", handle_greeting)
workflow.add_node("rag_qa", retrieve_and_answer)
workflow.add_node("lead_capture", capture_lead)

workflow.set_entry_point("classify")
workflow.add_conditional_edges("classify", route_intent)

workflow.add_edge("greeting", END)
workflow.add_edge("rag_qa", END)
workflow.add_edge("lead_capture", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# ----------------------------
# 7. CLI Interface
# ----------------------------
if __name__ == "__main__":
    print("🚀 AutoStream Chatbot (type 'quit' to exit)\n")

    thread_id = "user_123"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["quit", "exit"]:
            break

        events = app.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="values"
        )

        latest_msg = None

        for event in events:
            if "messages" in event:
                latest_msg = event["messages"][-1]

        if latest_msg:
            print(f"Agent: {latest_msg.content}")
        else:
            print("Agent: ⚠️ No response generated")