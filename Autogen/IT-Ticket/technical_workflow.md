# Technical Workflow & Architecture

## System Architecture

The system follows a **Hub-and-Spoke** architecture where the **Streamlit UI (`app.py`)** acts as the central controller (Hub), orchestrating interactions between the user and various specialized Agents (Spokes).

```mermaid
graph TD
    User[User] -->|Interacts| UI[Streamlit UI (app.py)]
    UI -->|1. Submit Question| Classifier[Classification Agent]
    Classifier -->|2. Return Category| UI
    UI -->|3. Query| KB[Knowledge Base Agent]
    KB -->|4. Return Answer + Confidence| UI
    UI -->|5. Low Confidence?| Web[Web Search Agent]
    Web -->|6. Return Search Results| UI
    UI -->|7. Display Answer| User
    User -->|8. Feedback (Satisfied/Escalate)| UI
    UI -->|9. Trigger| Notify[Notification Agent]
    Notify -->|10. Send Email| Email[Email System]
```

## Agent Interaction Flow

### 1. Initialization
- `app.py` initializes the AutoGen agents and the OpenAI client on startup.
- Session state is used to persist chat history and ticket status across re-runs.

### 2. Classification Phase
- **Input**: User's raw text query.
- **Process**: `Classification Agent` (LLM-based) analyzes the text.
- **Output**: One of [Network, Software, Access, Hardware, Performance, Security, Unknown].

### 3. Retrieval Phase
- **Primary**: `Knowledge Base Agent` generates an embedding for the query and searches the local vector store (simulated or ChromaDB).
- **Fallback**: If the best match score is below a threshold (e.g., 0.7), the `Web Search Agent` is invoked to find external information.

### 4. Resolution Phase
- The system presents the findings to the user.
- **Satisfied**: Ticket status -> `Resolved`. Notification sent.
- **Escalate**: Ticket status -> `Escalated`. Notification sent with full context (query, category, attempts).

## Data Flow & State Management

- **Session State (`st.session_state`)**:
    - `messages`: List of chat messages.
    - `ticket_status`: Current status (Open, Resolved, Escalated).
    - `category`: Detected issue category.
    - `user_query`: The original question.

## Error Handling Strategy

- **API Failures**: Try-except blocks around OpenAI and Search API calls. Graceful degradation (e.g., "Service unavailable, please contact support directly").
- **Empty Results**: If KB and Web Search fail, default to suggesting Escalation.
- **Input Validation**: Ensure user input is not empty before processing.
