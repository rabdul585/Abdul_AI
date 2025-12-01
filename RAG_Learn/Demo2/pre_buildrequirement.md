Here’s a ready-to-use prompt you can paste into ChatGPT / an LLM to get the full RAG app built for you:

````text
Role:
You are an expert AI engineer, Python developer, and Streamlit UI designer. You build production-ready Retrieval-Augmented Generation (RAG) apps with clean, modern, Google-like interfaces and well-structured, commented code.

Goal:
Build a complete RAG application using Streamlit as the front-end. The app must:
- Allow the user to upload one or more documents.
- Perform data loading, chunking, embedding, retrieval, and LLM-based question answering.
- If the answer is not present (or not well-supported) in the uploaded documents, gracefully fall back to web search.
- For every answer, compute and display:
  1. Relevance Score (0–1)
  2. Accuracy Score (0–1)
  3. Completeness Score (0–1)
- Clearly indicate whether the final answer is generated from:
  - Uploaded documents
  - Web search
  - Or a combination of both

Tech Stack Requirements:
- Language: Python
- UI Framework: Streamlit
- Vector Store: (You can choose a simple option like in-memory FAISS / Chroma / similar, and explain your choice)
- Embeddings: Use a common embedding model (e.g., OpenAI embeddings or any open-source alternative). Assume API keys are set via environment variables.
- LLM: Use a standard chat/completion model (OpenAI or open-source). Assume credentials/environment are already configured.
- Web Search:
  - Assume an existing helper function `search_web(query: str) -> str` (you can show a stub implementation and explain where to plug a real search API like Tavily/SerpAPI/etc.).
  - The RAG pipeline must call this function when document relevance is low or no answer is found in the local corpus.

Functional Requirements:
1. File Upload & Data Loading
   - Allow the user to upload at least PDF and TXT files (for example: `resume.pdf`).
   - Read and extract text from each uploaded file.
   - Merge all text into a corpus while preserving per-file metadata.

2. Chunking
   - Implement text chunking (e.g., by tokens or characters with overlap).
   - Parameters like chunk size and overlap should be configurable in the UI (with sensible defaults).
   - Store document metadata (file name, page index, chunk index) along with each chunk.

3. Embedding & Vector Store
   - Convert all chunks into embeddings.
   - Store them in an in-memory vector store for fast similarity search.
   - Provide a way to reset/rebuild the index when new files are uploaded.

4. Retrieval
   - Given a user query, retrieve the top-k most similar chunks from the uploaded documents.
   - Display retrieved chunks (optionally collapsible) for transparency/debugging.
   - Compute an aggregate "document relevance" indicator to decide whether doc-based answering is strong enough or if web search is required.

5. LLM Question Answering
   - Construct a prompt that:
     - Includes the user query.
     - Includes the top retrieved chunks.
     - Instructs the LLM to only answer from the provided context when possible and to explicitly say if the answer is not present.
   - Generate a concise, helpful answer.

6. Web Search Fallback & Fusion
   - If:
     - The retrieved chunks have low similarity scores, OR
     - The LLM indicates that the answer is not present in the documents,
     then:
       - Call `search_web(query)` to get external information.
       - Use a second LLM step to synthesize a final answer combining:
         - Retrieved document context (if any relevant), and
         - Web search results.
   - Always tag the answer with source information:
     - "Source: Uploaded documents"
     - "Source: Web search"
     - "Source: Uploaded documents + Web search"

7. Evaluation Metrics (Per Answer)
   For every user query and answer pair, compute three scores from 0 to 1:
   - Relevance Score (0–1):
     - How relevant are the retrieved chunks to the user query?
     - Use a simple heuristic (e.g., average similarity score normalized) OR a self-evaluation LLM call that reads query + context + answer and outputs a score.
   - Accuracy Score (0–1):
     - How factually correct is the answer with respect to the retrieved context and/or web search results?
     - Use an LLM-based self-evaluation: ask the model to rate accuracy 0–1 based on given context.
   - Completeness Score (0–1):
     - How fully does the answer address all parts of the user query?
     - Again, use an LLM-based self-evaluation with a clear rubric.

   Implementation detail:
   - Implement a helper function `evaluate_answer(query, context_snippets, web_snippets, answer) -> dict` that returns:
     ```python
     {
       "relevance": float,   # 0.0 to 1.0
       "accuracy": float,    # 0.0 to 1.0
       "completeness": float # 0.0 to 1.0
     }
     ```
   - Use a single LLM call with a rubric-style system prompt to produce these three scores.

8. Output Format to the User
   The UI must show, for each query:
   - **Final Answer** (nicely formatted)
   - **Source Label**:
     - Example: `Source: Uploaded documents`, `Source: Web search`, or `Source: Uploaded documents + Web search`
   - **Scores**:
     - Relevance: X.XX / 1.00
     - Accuracy: X.XX / 1.00
     - Completeness: X.XX / 1.00
   - Optional: summary of which chunks or web snippets were used.

9. Behavior Example (Few-Shot Requirement)
   - If the user uploads `resume.pdf` which contains educational qualifications and skillsets like "Python, AI, ML".
   - Then the user asks: "Who is Sachin Tendulkar?"
     - The system should:
       - Detect that the question is not answerable from the resume.
       - Use web search to answer "Who is Sachin Tendulkar?"
       - Answer using web information.
       - Still clearly state: `Source: Web search` (because the resume is not relevant).
   - If the user asks: "What are the AI-related skills mentioned in my resume?"
     - The system should:
       - Use the document (resume) to answer.
       - Optionally combine with web search if needed (though not necessary here).
       - Label: `Source: Uploaded documents`.

UI / UX Requirements (Google-like Look):
- Overall style:
  - Clean, minimal, "Google-like" white background.
  - Centered layout with a main card container for content.
  - Use a search-bar-style text input at the top for the user query:
    - Full-width (or large width) rounded rectangle.
    - Subtle shadow and hover effect.
  - Use a modern sans-serif font (e.g., default Streamlit + tweaks).
- Layout:
  - Top section:
    - App title (e.g., “RAG Q&A Assistant”) centered.
    - Short subtitle explaining: “Ask questions about your uploaded documents. If not found, I’ll search the web.”
  - Left / main column:
    - File uploader (allow multiple files).
    - Chunking and retrieval settings in an expandable “Advanced Settings” section.
    - Query input box and “Ask” button styled like a Google search bar and blue “Search” button.
  - Right / side or below:
    - A metrics card showing:
      - Relevance, Accuracy, Completeness as horizontal bars or badges.
    - A collapsible section showing:
      - Retrieved document chunks.
      - Web search snippets.
- Visual elements:
  - Use Streamlit components such as:
    - `st.container()`, `st.columns()`, `st.expander()`, `st.metric()` or `st.progress()` / `st.slider()` for score bars.
  - Smooth, simple feel—avoid clutter and unnecessary widgets.
- Make sure the UI is responsive and readable on standard desktop sizes.

Step-by-Step Approach (What You Should Output):
1. Briefly restate the problem and overall architecture.
2. Describe the RAG pipeline in clear stages:
   - Data loading
   - Chunking
   - Embedding
   - Vector store setup
   - Retrieval
   - LLM QA
   - Web search fallback & answer fusion
   - Evaluation (relevance, accuracy, completeness)
3. Describe the Streamlit UI layout and user flow.
4. Then provide the **full Streamlit Python code** in one file, including:
   - Imports
   - Helper functions:
     - Document loading and text extraction
     - Chunking
     - Embedding and vector store handling
     - Retrieval
     - Web search stub (`search_web`)
     - Answer generation
     - Evaluation function (`evaluate_answer`)
   - Streamlit `main()` app:
     - Layout and widgets
     - Handling file uploads and index building
     - Handling user queries and showing results/scores
5. Add clear inline comments explaining key logic and configuration points.
6. At the end, provide a short “How to Run” section (e.g., `pip install ...`, `streamlit run app.py`) and mention where to plug in real API keys/search APIs.

Constraints:
- Code must be syntactically correct and ready to paste into `app.py`.
- Avoid overly complex dependencies; keep it as simple and reproducible as possible.
- Be explicit about any assumptions (e.g., environment variables for API keys).
- Ensure every answer clearly shows source and evaluation scores as described.

Now, following all requirements above, produce:
1) A short architectural explanation, and
2) The complete Streamlit code implementation in a single Python file.
````
