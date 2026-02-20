
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client

# 1. Setup ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection(name="chunking_demo")
ollama_client = Client(host="http://localhost:11434")

# 2. Fairly big text as string
raw_text = """
Ramesh is a software engineer from Nepal who currently earns around $600 monthly working on AI projects.

He enjoys hiking in the Himalayas and often shares his experiences with friends.

His favorite programming language is Python, and he spends weekends contributing to open-source projects.

Recently, he started learning about machine learning and artificial intelligence, which he finds fascinating.

Ramesh believes that technology can help solve many of Nepal's challenges, especially in education and healthcare.

He is also passionate about teaching and regularly mentors students online.

In his free time, Ramesh likes to read books about history and science.

He dreams of building a tech startup that will create jobs for young people in his country.
"""
# 3. Split text into chunks using LangChain splitter


# Use LangChain's splitter with custom separator '\n' for paragraph splitting
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=0,
    separators=['\n']
)

chunks = [c.strip() for c in splitter.split_text(raw_text) if c.strip()]

print(f"Text split into {len(chunks)} chunks:")
for idx, c in enumerate(chunks):
    print(f"Chunk {idx}: '{c}'\n")

# 4. Embed and add to DB using Ollama nomic-embed-text
for i, text_chunk in enumerate(chunks):
    response = ollama_client.embed(model="nomic-embed-text", input=f"search_document: {text_chunk}")
    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[response["embeddings"][0]],
        documents=[text_chunk]
    )

# 5. User query: vectorize and retrieve
user_query = "Who earns $600?"
query_embed = ollama_client.embed(model="nomic-embed-text", input=f"query: {user_query}")["embeddings"][0]
results = collection.query(query_embeddings=[query_embed], n_results=1)

print("Most relevant chunk:")
for doc in results["documents"][0]:
    print(doc)
