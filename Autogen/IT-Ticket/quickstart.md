# Quickstart Guide - AI IT Support Ticketing System

## Prerequisites
- Python 3.8+
- OpenAI API Key

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**:
   - Copy `.env.example` to `.env`:
     ```bash
     # Windows (PowerShell)
     copy .env.example .env
     # Mac/Linux
     cp .env.example .env
     ```
   - Open `.env` and add your `OPENAI_API_KEY`.
   - (Optional) Add `SERPAPI_API_KEY` for web search.

## Running the Application

Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. **Submit a Question**: Type your IT issue in the chat box (e.g., "My wifi is not working").
2. **View Analysis**: The system will classify the issue and search for solutions.
3. **Resolve or Escalate**:
   - Click **✅ Yes, Resolved** if the answer helped.
   - Click **🚨 No, Escalate** to create a support ticket.

## Troubleshooting

- **OpenAI Error**: Ensure your API key is correct in `.env`.
- **Module Not Found**: Run `pip install -r requirements.txt` again.
