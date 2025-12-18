# 🤖 AI IT Support Ticketing System

An intelligent, multi-agent IT support system built with **Streamlit** and **AutoGen**. It automatically classifies user issues, retrieves solutions from a Knowledge Base, performs web searches for unknown issues, and handles ticket escalation with email notifications.

## 🏗️ System Architecture

The system follows a **Hub-and-Spoke** architecture where the Streamlit UI acts as the central orchestrator.

```mermaid
graph TD
    User[User] -->|1. Enters Query| UI[Streamlit App (app.py)]
    UI -->|2. Classify| Classifier[Classifier Agent]
    Classifier -->|3. Return Category| UI
    UI -->|4. Search KB| KB[Knowledge Base Agent]
    KB -->|5. Return Answer| UI
    UI -->|6. If KB Fails (Fallback)| Web[Web Search Agent]
    Web -->|7. Return Web Results| UI
    UI -->|8. Display Solution| User
    User -->|9. Feedback (Resolved/Escalate)| UI
    UI -->|10. Send Email| Notify[Notification Agent]
```

### Core Components

1.  **`app.py`**: The main entry point. Manages session state, user interaction, and agent orchestration.
2.  **`agent/classifier_agent.py`**: Uses LLM (GPT-4o-mini) to categorize issues (Network, Software, Hardware, etc.).
3.  **`agent/knowledge_base_agent.py`**: Searches a local text-based KB (`KBDocs/knowledge_base.txt`) for relevant troubleshooting steps.
4.  **`agent/web_search_agent.py`**: Uses **SerpApi** to search Google when the KB doesn't have an answer.
5.  **`agent/notification_agent.py`**: Sends real emails via SMTP for ticket resolution or escalation.

---

## 🚀 End-to-End Flow

1.  **User Login**: User enters Name and Email.
2.  **Issue Submission**: User types an IT problem (e.g., "Wifi not working").
3.  **Classification**: The system identifies the category (e.g., "Network").
4.  **Resolution Retrieval**:
    *   **Step A**: Search Knowledge Base. If a match is found, show it.
    *   **Step B**: If KB returns "No relevant article", automatically search the Web.
5.  **User Action**:
    *   **✅ Resolved**: Ticket closed, confirmation email sent.
    *   **🚨 Escalate**: Ticket escalated, support team notified via email.

---

## 🛠️ Setup & Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment**:
    Create a `.env` file with your keys:
    ```ini
    OPENAI_API_KEY=sk-...
    SERPAPI_API_KEY=...
    SMTP_SERVER=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USERNAME=your_email@gmail.com
    SMTP_PASSWORD=your_app_password
    ```

3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

---

## 💡 Tips for Improvement

1.  **Vector Database**: Replace the text-based KB with a real Vector DB (ChromaDB or Pinecone) for semantic search capabilities.
2.  **Agentic Workflow**: Use AutoGen's `GroupChat` to let agents converse with each other (e.g., Classifier talks to KB directly) instead of `app.py` orchestrating everything.
3.  **Ticket Database**: Integrate a real database (SQLite/PostgreSQL) to store ticket history instead of just session state.
4.  **Advanced RAG**: Implement "Hybrid Search" (Keyword + Semantic) for better KB retrieval accuracy.
5.  **Admin Dashboard**: Create a separate page for IT admins to view and manage escalated tickets.

## 📂 Project Structure

```
├── agent/                  # Agent implementations
│   ├── classifier_agent.py
│   ├── knowledge_base_agent.py
│   ├── notification_agent.py
│   └── web_search_agent.py
├── KBDocs/                 # Knowledge Base data
│   └── knowledge_base.txt
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── test_serp.py            # Utility to test SerpApi
└── .env                    # API keys and config
```
