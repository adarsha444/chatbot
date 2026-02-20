import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client
import chromadb
import ollama
chat_bot = ollama.Client(host="http://localhost:11434")

# 1. Initialize the local Vector Database (ChromaDB)
client = chromadb.PersistentClient()
remote_client = Client(host=f'http://localhost:11434')

#collection = client.get_or_create_collection(name="simple_knowledge")

collection = client.get_or_create_collection(name="simple_knowledge")
counter=0
if os.path.exists('counter.txt'):
    with open('counter.txt', 'r') as f:
        counter = int(f.read())
        collection=client.get_or_create_collection(name=f"simple_knowledge")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
    separators=[".", "\n"]
)
#current_count = collection.count()
#if not collection.count():
     #print("Creating a new database...")
with open('articles.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < counter:
            print(f"Skipping line {i}")       
            continue  
        print(f"Adding line {i}")  
        content = json.loads(line)['content']

        chunks = [c.strip() for c in splitter.split_text(content) if c.strip()]
        for j, c in enumerate(chunks):
            #response = remote_client.embed(model='nomic-embed-text', input=content)
            response = remote_client.embed(model='nomic-embed-text', input=f"search_document: {c}")
        
            embedding = response['embeddings'][0]
            collection.add(
              ids=[f"id_{i}_{j}"],
              embeddings=[embedding],
              documents=[c],
             metadatas=[{"line": j}]
            )
print("Database built successfully!")

# 3. Test Retrieval
query = "Who finishes off in style in the 2011 World Cup final?"
query_embed = remote_client.embed(model='nomic-embed-text', input=query)['embeddings'][0]
# query_embed = remote_client.embed(
#     model='nomic-embed-text', 
#     input=f"search_query: {query}"
# )['embeddings'][0]
while True:
    user_input = input("How may I assist you? ")
    query_embd=ollama.embed(model="nomic-embed-text", input=f"query: {user_input}")["embeddings"][0]
    results = collection.query(query_embeddings=[query_embd], n_results=2)
    
    retrieved_docs = results['documents'][0]
    context = "\n\n".join(retrieved_docs)


    prompt = f"""You are a helpful assistant. Answer the question based on the context provided. Use the information in the context to form your answer. If context does not have enough information just say "I don't know"

    Context: {context}

    Question: {query}

    Answer:"""

    response = chat_bot.generate(
            model="qwen3:4b-instruct-2507-q4_K_M",
            prompt=prompt,
            options={
                "temperature": 0.1
            }
        )

    answer = response['response']

    print(answer)