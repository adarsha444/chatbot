from ollama import Client
import json
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------------
# CONFIG
# ----------------------------
OLLAMA_HOST = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text:latest"
COLLECTION_NAME = "articles_demo"

# ----------------------------
# CLIENTS
# ----------------------------
chroma_client = chromadb.PersistentClient()
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

ollama_client = Client(host=OLLAMA_HOST)

# ----------------------------
# CHUNKING
# ----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separators=[".", "\n"]
)

# ----------------------------
# BUILD DATABASE
# ----------------------------
if collection.count() <= 0:

    print("Reading articles.jsonl and generating embeddings...\n")

    total_lines = 0
    total_chunks_processed = 0
    total_chunks_skipped = 0

    with open("articles.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):

            total_lines += 1
            print(f"Processing line {total_lines}...")

            article = json.loads(line)
            content = article.get("content", "")
            title = article.get("title", "Untitled")

            raw_chunks = splitter.split_text(content)

            for j, chunk in enumerate(raw_chunks):

                chunk = chunk.strip()

                if len(chunk) <= 15:
                    total_chunks_skipped += 1
                    continue

                total_chunks_processed += 1

                print(f"   → Embedding chunk {j} (Processed: {total_chunks_processed}, Skipped: {total_chunks_skipped})")

                resp = ollama_client.embed(
                    model=EMBED_MODEL,
                    input=f"search_document: {chunk}"
                )

                embedding = resp["embeddings"][0]

                collection.add(
                    ids=[f"article_{i}_chunk_{j}"],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"title": title}],
                )

    print("\n----------------------------")
    print("Database build completed!")
    print(f"Total lines read       : {total_lines}")
    print(f"Chunks processed       : {total_chunks_processed}")
    print(f"Chunks skipped (small) : {total_chunks_skipped}")
    print("----------------------------\n")

else:
    print("Database already exists. Skipping build.")

# ----------------------------
# QUERY SECTION
# ----------------------------
query = "are there any predicted hindrance for upcoming election ?"

query_embed = ollama_client.embed(
    model=EMBED_MODEL,
    input=f"query: {query}"
)["embeddings"][0]

results = collection.query(query_embeddings=[query_embed], n_results=1)

print(f"\nQuestion: {query}")
print(f'\nTitle: {results["metadatas"][0][0]["title"]}')
print(results["documents"][0][0])
## ask to the user to input a query and show the result 
## query from the user and show the result in a loop until the user wants to exit
#