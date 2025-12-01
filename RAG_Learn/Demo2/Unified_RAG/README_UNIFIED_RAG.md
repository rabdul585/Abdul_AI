# 🚀 Unified RAG System

A comprehensive Streamlit application that integrates **5 powerful RAG (Retrieval-Augmented Generation) strategies** into one intelligent system.

## ✨ Features

### 🔍 **Corrective RAG**
- **Evaluates quality** of retrieved context automatically
- **Refines queries** when context quality is poor
- **Multiple attempts** (configurable 1-5) to get better results
- **Quality metrics**: Relevance, Completeness, Accuracy, Specificity
- **Best for**: Quality-critical queries requiring accurate information

### 🔄 **Fallback RAG**
- **5 fallback levels** ensure robust retrieval
- Progressively increases search scope (k=5, 10, 15, 20, 25)
- Adjustable confidence thresholds per level
- **Best for**: Difficult-to-answer queries that need broader context

### 🌐 **Web Search RAG**
- **Combines internal knowledge + web search** intelligently
- Uses SERPAPI for real-time web results
- Can use internal only, web only, or combined
- **Best for**: Current events, external information, latest news

### 🎯 **Adaptive RAG**
- **Adjusts to query complexity** (Simple, Moderate, Complex)
- **Detects user level** (Beginner, Intermediate, Expert)
- **Tailors response style** based on audience
- **Best for**: Varied audience levels and query types

### 🚀 **Unified System**
- **Auto-selects best method** for each query
- Analyzes query characteristics
- Routes to optimal RAG strategy
- **Best for**: General purpose use, maximum flexibility

## 🎨 Beautiful UI Features

- **Gradient-based design** with modern aesthetics
- **Color-coded RAG types** with distinct badges
- **Dynamic parameter controls** that change based on selected RAG type
- **Real-time statistics** tracking usage of each strategy
- **Interactive chat interface** with metadata display
- **Responsive layout** with sidebar configuration

## 🛠️ Installation

1. **Install dependencies**:
```bash
pip install streamlit langchain langchain-openai langchain-community langchain-text-splitters python-dotenv requests chromadb
```

2. **Set up environment variables** in `.env`:
```env
OPENAI_API_KEY=your-openai-api-key-here
SERPAPI_API_KEY=your-serpapi-key-here
```

3. **Run the application**:
```bash
streamlit run unified_rag_system.py
```

## 📖 Usage Guide

### 1. **Initialize Sample Data**
- Click "🔄 Initialize Sample Data" in the sidebar
- Loads 10 sample documents about machine learning
- Creates a Chroma vectorstore with OpenAI embeddings

### 2. **Select RAG Type**
Choose from 5 strategies:
- **Corrective RAG**: For quality-focused retrieval
- **Fallback RAG**: For robust multi-level search
- **Web Search RAG**: For external information
- **Adaptive RAG**: For audience-tailored responses
- **Unified System**: For automatic method selection

### 3. **Adjust Parameters**
Each RAG type has unique parameters:

#### Corrective RAG
- **Max Correction Attempts**: 1-5 (default: 3)
- **Quality Threshold**: POOR/AVERAGE/GOOD/EXCELLENT

#### Fallback RAG
- Pre-configured 5 levels (view in sidebar)
- Automatic threshold adjustment

#### Web Search RAG
- **Use Internal Knowledge**: Toggle internal search
- **Use Web Search**: Toggle web search
- **Combine Results**: Merge both sources

#### Adaptive RAG
- Auto-adjusts based on query analysis
- No manual configuration needed

#### Unified System
- Fully automatic method selection
- No manual configuration needed

### 4. **Ask Questions**
Type your question in the chat interface and press "Send 🚀"

**Example Questions**:
- "What is overfitting in machine learning?" (Corrective RAG)
- "Explain neural networks to a beginner" (Adaptive RAG)
- "What are the latest AI trends?" (Web Search RAG)
- "How does gradient descent work?" (Fallback RAG)
- "Tell me about deep learning" (Unified System)

### 5. **View Results**
- **Chat messages** show the response
- **Method badge** indicates which RAG strategy was used
- **Metadata expander** reveals detailed metrics
- **Statistics** track usage of each strategy

## 🔧 Configuration

### Model Settings
- **Model**: `gpt-4o-mini` (default, configurable)
- **Temperature**: 0.0-1.0 (default: 0.7)

### API Keys Required
- **OPENAI_API_KEY**: For LLM and embeddings
- **SERPAPI_API_KEY**: For web search (optional, only needed for Web Search RAG)

Get your SERPAPI key from: https://www.searchapi.io/

## 📊 How Each RAG Strategy Works

### Corrective RAG Flow
```
Query → Retrieve Context → Evaluate Quality
  ↓
  If Quality < Threshold:
    Refine Query → Retrieve Again → Evaluate
  ↓
  Generate Answer with Best Context
```

### Fallback RAG Flow
```
Query → Level 1 (k=5, threshold=0.7)
  ↓ (if insufficient)
  Level 2 (k=10, threshold=0.6)
  ↓ (if insufficient)
  Level 3 (k=15, threshold=0.5)
  ↓ (if insufficient)
  Level 4 (k=20, threshold=0.4)
  ↓ (if insufficient)
  Level 5 (k=25, threshold=0.3)
  ↓
  Generate Answer
```

### Web Search RAG Flow
```
Query → Internal Search (if enabled)
  ↓
  Web Search (if enabled)
  ↓
  Combine Results (if both available)
  ↓
  Generate Comprehensive Answer
```

### Adaptive RAG Flow
```
Query → Analyze Complexity & User Level
  ↓
  Adjust k (3/5/10 based on complexity)
  ↓
  Adjust Response Style (Beginner/Intermediate/Expert)
  ↓
  Generate Tailored Answer
```

### Unified System Flow
```
Query → Analyze Query Characteristics
  ↓
  Auto-Select Best Method:
    - Corrective (quality-critical)
    - Fallback (difficult queries)
    - Web Search (external info)
    - Adaptive (varied complexity)
  ↓
  Route to Selected Method
  ↓
  Generate Answer
```

## 🎯 Use Cases

### Corrective RAG
- Medical/Legal queries requiring accuracy
- Technical documentation search
- Research paper analysis

### Fallback RAG
- Complex multi-faceted questions
- Queries with ambiguous terms
- Broad exploratory searches

### Web Search RAG
- Current events and news
- Product information and reviews
- Latest technology trends

### Adaptive RAG
- Educational content for different levels
- Customer support with varied expertise
- Tutorial and documentation generation

### Unified System
- General chatbots
- Multi-purpose assistants
- Unknown query types

## 🎨 UI Customization

The application features:
- **Gradient headers** with purple-pink theme
- **Color-coded badges** for each RAG type
- **Hover effects** on cards and buttons
- **Smooth transitions** and animations
- **Responsive design** for different screen sizes

## 📈 Statistics Tracking

The sidebar displays real-time usage statistics:
- **Corrective**: Number of corrective RAG queries
- **Fallback**: Number of fallback RAG queries
- **Web Search**: Number of web search RAG queries
- **Adaptive**: Number of adaptive RAG queries
- **Unified**: Number of unified system queries

## 🔒 Security

- API keys loaded from `.env` file (not hardcoded)
- Environment variables used for sensitive data
- No API keys exposed in UI

## 🚀 Performance Tips

1. **Use Unified System** for automatic optimization
2. **Choose specific RAG type** when you know the query characteristics
3. **Initialize sample data** once and reuse
4. **Clear chat history** periodically to free memory
5. **Adjust temperature** based on creativity needs (lower = more focused)

## 🐛 Troubleshooting

### "Please initialize sample data first!"
- Click "🔄 Initialize Sample Data" in sidebar

### "SearchAPI request failed"
- Check SERPAPI_API_KEY in `.env`
- Verify API key is valid
- Web Search RAG requires valid SERPAPI key

### "OpenAI API error"
- Check OPENAI_API_KEY in `.env`
- Verify API key has sufficient credits
- Check internet connection

## 📝 Example Queries by RAG Type

### Corrective RAG
```
"What are the side effects of overfitting in machine learning?"
"Explain the mathematical foundation of neural networks"
```

### Fallback RAG
```
"What is the relationship between AI, ML, and deep learning?"
"How do transformers work in natural language processing?"
```

### Web Search RAG
```
"What are the latest developments in GPT models?"
"Current trends in artificial intelligence 2024"
```

### Adaptive RAG
```
"Explain gradient descent" (auto-detects user level)
"What is a neural network?" (adjusts complexity)
```

### Unified System
```
"Tell me about machine learning" (auto-selects best method)
"How does AI work?" (routes intelligently)
```

## 🎓 Learning Resources

- **LangChain Documentation**: https://python.langchain.com/
- **OpenAI API**: https://platform.openai.com/docs
- **Streamlit Docs**: https://docs.streamlit.io/
- **RAG Concepts**: https://www.pinecone.io/learn/retrieval-augmented-generation/

## 🤝 Contributing

Feel free to extend this system with:
- Additional RAG strategies
- Custom evaluation metrics
- New data sources
- Enhanced UI components

## 📄 License

MIT License - Feel free to use and modify!

## 🙏 Acknowledgments

Built with:
- **Streamlit** - Beautiful web apps
- **LangChain** - LLM framework
- **OpenAI** - GPT models
- **Chroma** - Vector database
- **SERPAPI** - Web search

---

**Made with ❤️ by Kanda**

🔗 **Quick Start**: `streamlit run unified_rag_system.py`
