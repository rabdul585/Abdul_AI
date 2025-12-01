import os, json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from neo4j import GraphDatabase

load_dotenv()

NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vector_db = PGVector(
    connection_string=NEON_DATABASE_URL,
    collection_name=VECTOR_COLLECTION,
    embedding_function=embeddings
)

driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

TRIPLETS_PROMPT = """
Extract factual triplets ONLY from the text.
Return JSON list:
[
 {"subject":"...","predicate":"...","object":"..."}
]
Rules:
- Only explicit facts
- If none, return []
Text:
{chunk}
"""

def create_constraints(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")

def insert_triplet(tx, s, p, o):
    tx.run("""
    MERGE (a:Entity {name:$s})
    MERGE (b:Entity {name:$o})
    MERGE (a)-[:REL {predicate:$p}]->(b)
    """, s=s, p=p, o=o)

def main():
    docs = vector_db.similarity_search(" ", k=5000)

    with driver.session() as session:
        session.execute_write(create_constraints)

    total = 0
    for d in docs:
        chunk = d.page_content

        res = llm.invoke(TRIPLETS_PROMPT.format(chunk=chunk))
        try:
            triplets = json.loads(res.content)
        except:
            continue

        with driver.session() as session:
            for t in triplets:
                s, p, o = t.get("subject"), t.get("predicate"), t.get("object")
                if s and p and o:
                    session.execute_write(insert_triplet, s.strip(), p.strip(), o.strip())
                    total += 1

    print(f"✅ KG built with {total} triplets in Neo4j.")

if __name__ == "__main__":
    main()
