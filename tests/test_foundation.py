from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()

# Test 1: Gemini LLM call
print("Test 1: Gemini LLM call")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
response = llm.invoke("Say hello in one sentence.")
print(f"Response: {response.content}")
print("PASS: Gemini works!\n")

# Test 2: ChromaDB write + query
print("Test 2: ChromaDB write + query")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection("test_qa", embedding_function=ef)

collection.add(
    documents=[
        "Python lists are mutable, tuples are immutable.",
        "Big O notation describes algorithm complexity.",
        "STAR method: Situation, Task, Action, Result.",
    ],
    ids=["1", "2", "3"],
)

results = collection.query(query_texts=["data structures in Python"], n_results=2)
for doc in results["documents"][0]:
    print(f"  - {doc}")
print("PASS: ChromaDB works!")
