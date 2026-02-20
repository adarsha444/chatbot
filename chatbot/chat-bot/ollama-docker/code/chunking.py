from ollama import Client
import chromadb

# 1. Setup the Database
client = chromadb.Client()
remote_client = Client(host=f'http://localhost:11434')
collection = client.get_or_create_collection(name="chunking_demo")

# 2. The Raw Data (A single long sentence/paragraph)
raw_text = "Ramesh is a software engineer from Nepal who currently earns around $600 monthly working on AI projects."

# 3. Simple Chunking Logic
# We split by characters for this demo to show how words get separated
chunk_size = 30  
overlap = 10
chunks = []

for i in range(0, len(raw_text), chunk_size - overlap):
    chunk = raw_text[i : i + chunk_size]
    chunks.append(chunk)

print(f"Original text split into {len(chunks)} chunks:")
for idx, c in enumerate(chunks):
    print(f"Chunk {idx}: '{c}'")

# 4. Embedding and Adding to DB
print("\nEmbedding chunks...")
for i, text_chunk in enumerate(chunks):
    response = remote_client.embed(model='nomic-embed-text', input=f'search_document: {text_chunk}')
    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[response['embeddings'][0]],
        documents=[text_chunk]
    )

# 5. The "Incomplete Context" Search
query = "Who earns $600?"
query_embed = remote_client.embed(model='nomic-embed-text', input=f'query: {query}')['embeddings'][0]

# We ask for the single most similar chunk
results = collection.query(
    query_embeddings=[query_embed],
    n_results=2
)

full_context = "\n".join(results['documents'][0])

print(f"Combined Context:\n{full_context}")