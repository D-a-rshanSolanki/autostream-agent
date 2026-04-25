# AutoStream: Social-to-Lead Agentic Workflow

An AI-powered conversational agent built for AutoStream, designed to answer product queries using RAG, identify high-intent users, and capture lead information.

## How to run the project locally

1. Clone the repository and navigate to the project directory.
2. Create a virtual environment and activate it:
   `python -m venv venv`
   `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
3. Install dependencies:
   `pip install -r requirements.txt`
4. Create a `.env` file in the root directory and add your Google Gemini API key:
   `GOOGLE_API_KEY=your_api_key_here`
5. Run the agent:
   `python agent.py`

## Architecture Explanation

This agent is built using **LangGraph** because it provides robust state management, which is essential for retaining memory across conversation turns and ensuring tools are not triggered prematurely. 

The architecture uses a `StateGraph` with a defined `AgentState` containing the message history, current intent, and extracted lead data. 
1. **Routing:** Every user message first hits the `intent_classifier` node.
2. **Execution:** Based on the classification, LangGraph conditionally routes the flow to either a casual greeting handler, a RAG-powered knowledge retrieval node, or the lead capture node. 
3. **State Management:** By using LangGraph's `MemorySaver` as a checkpointer, the agent easily maintains context across multiple interactions, allowing it to remember previously provided lead details (like a name) while asking for missing ones (like an email).

## WhatsApp Deployment Integration

To integrate this agent with WhatsApp using Webhooks:
1. **Meta App Setup:** Create an app in the Meta Developer Portal and configure the WhatsApp Business API.
2. **Webhook Endpoint:** Expose a POST endpoint in the Python application (using FastAPI or Flask) that Meta can send incoming WhatsApp messages to.
3. **Payload Processing:** When a message hits the webhook, extract the user's phone number (to use as a unique `thread_id` in LangGraph for state management) and the message text.
4. **Agent Invocation:** Pass the text to the LangGraph `app.invoke()` function.
5. **Response Delivery:** Capture the agent's output and make an HTTP POST request back to the WhatsApp Graph API to deliver the response to the user's phone.