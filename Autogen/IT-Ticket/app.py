import streamlit as st
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv

# Import Agents
from agent.classifier_agent import ClassifierAgent
from agent.knowledge_base_agent import KnowledgeBaseAgent
from agent.web_search_agent import WebSearchAgent
from agent.notification_agent import NotificationAgent

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="AI IT Support", page_icon="🤖", layout="centered")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "ticket_id" not in st.session_state:
    st.session_state.ticket_id = str(uuid.uuid4())[:8]
if "ticket_status" not in st.session_state:
    st.session_state.ticket_status = "Open"  # Open, Resolved, Escalated
if "category" not in st.session_state:
    st.session_state.category = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "user_info" not in st.session_state:
    st.session_state.user_info = {"name": "", "email": ""}
if "user_details_submitted" not in st.session_state:
    st.session_state.user_details_submitted = False

# Initialize Agents (Cached to avoid reloading on every rerun)
@st.cache_resource
def get_agents():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not found in .env file.")
        return None, None, None, None
    
    classifier = ClassifierAgent(api_key=api_key)
    kb = KnowledgeBaseAgent(api_key=api_key)
    web = WebSearchAgent(api_key=os.getenv("SERPAPI_API_KEY"))
    notifier = NotificationAgent()
    
    return classifier, kb, web, notifier

classifier_agent, kb_agent, web_agent, notification_agent = get_agents()

# UI Header
st.title("🤖 AI IT Support System")

# User Details Form
if not st.session_state.user_details_submitted:
    st.markdown("### 👋 Welcome! Please enter your details to start.")
    with st.form("user_details_form"):
        name = st.text_input("Name")
        email = st.text_input("Email Address")
        submit_button = st.form_submit_button("Start Support Session")
        
        if submit_button:
            if name and email:
                st.session_state.user_info = {"name": name, "email": email}
                st.session_state.user_details_submitted = True
                st.rerun()
            else:
                st.error("Please enter both Name and Email.")
    st.stop() # Stop execution here until details are submitted

# Main Interface (After Login)
status_color = {
    "Open": "blue",
    "Resolved": "green",
    "Escalated": "red"
}
st.markdown(f"**User:** {st.session_state.user_info['name']} | **Ticket ID:** `{st.session_state.ticket_id}` | **Status:** :{status_color.get(st.session_state.ticket_status, 'grey')}[{st.session_state.ticket_status}]")

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Handling
if st.session_state.ticket_status == "Open" and not st.session_state.processing_complete:
    user_input = st.chat_input("Describe your IT issue...")
    
    if user_input:
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # 2. Classification
        with st.status("🔍 Identifying issue type...", expanded=True) as status:
            category = classifier_agent.classify(user_input)
            st.session_state.category = category
            status.update(label=f"Issue Detected: **{category}**", state="complete")
            
        # 3. Retrieval (KB -> Web)
        with st.spinner("Searching knowledge base..."):
            answer, confidence = kb_agent.search(user_input, category)
            source = "Knowledge Base"
            
            # Check if KB failed to find an answer (ignoring score, checking content as requested)
            if "No relevant KB article found" in answer:
                with st.spinner("Searching web..."):
                    web_answer = web_agent.search(user_input, category)
                    # Always use web answer if KB failed
                    answer = web_answer
                    source = "Web Search"
        
        # 4. Display Solution
        response_text = f"**Category:** {category}\n**Source:** {source}\n\n{answer}"
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)
            
        st.session_state.processing_complete = True
        st.rerun()

# Feedback / Resolution Phase
if st.session_state.processing_complete and st.session_state.ticket_status == "Open":
    st.markdown("---")
    st.subheader("Did this solve your issue?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Yes, Resolved", use_container_width=True):
            st.session_state.ticket_status = "Resolved"
            notification_agent.send_notification(
                st.session_state.ticket_id,
                "Resolved",
                {
                    "category": st.session_state.category,
                    "query": st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else "",
                    "action": "User confirmed resolution",
                    "timestamp": datetime.now().isoformat()
                },
                user_email=st.session_state.user_info["email"]
            )
            st.success("Ticket closed successfully! Confirmation email sent.")
            st.rerun()
            
    with col2:
        if st.button("🚨 No, Escalate", use_container_width=True):
            st.session_state.ticket_status = "Escalated"
            notification_agent.send_notification(
                st.session_state.ticket_id,
                "Escalated",
                {
                    "category": st.session_state.category,
                    "query": st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else "",
                    "action": "User requested escalation",
                    "timestamp": datetime.now().isoformat()
                },
                user_email=st.session_state.user_info["email"]
            )
            st.error("Ticket escalated to support team. An email has been sent to you.")
            st.rerun()

# Post-Resolution/Escalation View
if st.session_state.ticket_status != "Open":
    if st.button("Start New Ticket"):
        # Keep user info, reset ticket details
        user_info = st.session_state.user_info
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.user_info = user_info
        st.session_state.user_details_submitted = True
        st.rerun()
