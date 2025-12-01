"""Simple self-RAG example using ChatOpenAI and FAISS.

This implements the pattern shown in the screenshot: ask the LLM directly first,
and if the answer looks low-confidence, run a retrieval-augmented query.

Notes:
- Ensure `my_notes.txt` exists in this folder (or change the path).
- Set `OPENAI_API_KEY` in your environment before running.
"""

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os


def build_rag_index(doc_path: str = r"D:\Abdul_AI\RAG_Learn\Self_RAG\my_notes.txt"):
    loader = TextLoader(doc_path)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def make_qa_chain(vectorstore):
	retriever = vectorstore.as_retriever()
	llm = ChatOpenAI(temperature=0)
	qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
	return llm, qa


def self_rag_query(llm, qa, question: str):
	print("First attempt without retrieval:")

	# Try using the same llm.predict style as in the screenshot.
	try:
		first_answer = llm.predict(f"Q: {question}\nA:")
	except Exception:
		# Fallback for newer LangChain versions (callable llm)
		try:
			resp = llm(f"Q: {question}\nA:")
			# try to extract text
			if isinstance(resp, str):
				first_answer = resp
			else:
				# attempt to get text from generation structure
				first_answer = getattr(resp, "text", str(resp))
		except Exception:
			first_answer = ""

	print(first_answer)

	if ("I'm not sure" in first_answer) or (len(first_answer or "") < 30):
		print("Low confidence. Retrieving context and trying again...")
		improved_answer = qa.run(question)
		return improved_answer
	else:
		return first_answer


if __name__ == "__main__":
	# Ensure API key is set (OpenAI embeddings + ChatOpenAI require it)
	if not os.environ.get("OPENAI_API_KEY"):
		print("Warning: OPENAI_API_KEY not set in environment. Set it before running.")

	# Build vector index from a local file named my_notes.txt
	ds = build_rag_index("my_notes.txt")
	llm, qa = make_qa_chain(ds)

	response = self_rag_query(llm, qa, "What is the capital of France?")
	print("\nFinal Answer:", response)
