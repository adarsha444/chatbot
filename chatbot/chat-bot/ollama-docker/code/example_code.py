import json
from pathlib import Path

import chromadb
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client

# ==============================
# PHASE 1: CONFIGURATION
# ==============================
# Ollama server URL (must be running locally).
OLLAMA_HOST = "http://172.16.9.90:11434"
# Embedding model used for vector search.
EMBED_MODEL = "nomic-embed-text"
# Chat model used to generate the final answer.
CHAT_MODEL = "qwen2.5:3b"
# Chroma collection name.
COLLECTION_NAME = "simple_knowledge"
# Number of chunks to retrieve for each question.
TOP_K = 3
# Confidence threshold for accepting retrieved chunks
CONFIDENCE_THRESHOLD = 0.5

# Keep all file paths relative to this script file.
BASE_DIR = Path(__file__).resolve().parent
ARTICLES_FILE = BASE_DIR / "articles.jsonl"
COUNTER_FILE = BASE_DIR / "counter.txt"
CHROMA_DIR = BASE_DIR / "chroma"

# ==============================
# PHASE 2: CLIENTS + COLLECTION
# ==============================
# Single Ollama client for both embedding and answer generation.
ollama_client = Client(host=OLLAMA_HOST)
# Persistent Chroma storage directory.
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
# Create/load vector collection.
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ==============================
# PHASE 3: TEXT SPLITTING SETUP
# ==============================
# Split long article content into smaller chunks for embedding.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
    separators=[".", "\n"],
)


# ==============================
# PHASE 4: HELPER FUNCTIONS
# ==============================
def load_counter() -> int:
    # Read the last processed line index from counter file (resume support).
    if not COUNTER_FILE.exists():
        return 0
    try:
        raw = COUNTER_FILE.read_text(encoding="utf-8").strip()
        return int(raw) if raw else 0
    except ValueError:
        return 0


def save_counter(next_line_index: int) -> None:
    # Save the next line index to process.
    COUNTER_FILE.write_text(str(next_line_index), encoding="utf-8")


def load_and_embed_articles_lazy() -> None:
    # Lazy load: only embed articles that haven't been processed yet.
    if not ARTICLES_FILE.exists():
        raise FileNotFoundError(f"Missing file: {ARTICLES_FILE}")

    start_line = load_counter()
    loaded_count = 0

    with ARTICLES_FILE.open("r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            # Skip lines already processed.
            if i < start_line:
                continue

            # Parse one JSON line.
            article = json.loads(line)
            content = article.get("content", "").strip()
            title = article.get("title", "Untitled")

            if not content:
                save_counter(i + 1)
                continue

            # Split article into chunks.
            chunks = [c.strip() for c in splitter.split_text(content) if c.strip()]

            # Batch embed all chunks at once for faster processing
            if chunks:
                batch_inputs = [f"search_document: {chunk}" for chunk in chunks]
                embed_response = ollama_client.embed(
                    model=EMBED_MODEL,
                    input=batch_inputs,  # Send all chunks at once
                )
                embeddings = embed_response["embeddings"]

                # Upsert all chunks in batch
                ids = [f"id_{i}_{j}" for j in range(len(chunks))]
                collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=[{"line": i, "chunk": j, "title": title} for j in range(len(chunks))],
                )

            save_counter(i + 1)
            loaded_count += 1
            print(f"Loaded article {i}: {title}")

    if loaded_count > 0:
        print(f"Loaded {loaded_count} new article(s).")
    else:
        print("All articles already loaded.")


def retrieve_context(question: str, n_results: int = TOP_K) -> tuple:
    # Convert user question to embedding vector.
    query_embedding = ollama_client.embed(
        model=EMBED_MODEL,
        input=f"query: {question}",
    )["embeddings"][0]

    # Retrieve top matching chunks from vector DB.
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    retrieved_docs = results.get("documents", [[]])[0]
    retrieved_distances = results.get("distances", [[]])[0]
    retrieved_ids = results.get("ids", [[]])[0]

    # Calculate confidence scores (convert distance to similarity)
    chunks_with_confidence = []
    for doc_id, distance, doc in zip(retrieved_ids, retrieved_distances, retrieved_docs):
        similarity = 1 / (1 + distance)  # Convert L2 distance to similarity
        confidence = max(0, similarity)
        chunks_with_confidence.append({
            'text': doc,
            'id': doc_id,
            'confidence': confidence
        })
    
    # Filter by confidence threshold
    confirmed_chunks = [c for c in chunks_with_confidence if c['confidence'] >= CONFIDENCE_THRESHOLD]
    if not confirmed_chunks:
        confirmed_chunks = chunks_with_confidence  # Use all if none meet threshold
    
    # Calculate average confidence
    avg_confidence = sum(c['confidence'] for c in confirmed_chunks) / len(confirmed_chunks) if confirmed_chunks else 0
    
    # Merge docs into one context block for prompt.
    context = "\n\n".join(chunk['text'] for chunk in confirmed_chunks if chunk['text'])
    
    return context, avg_confidence, len(confirmed_chunks)


def generate_answer(question: str, context: str, confidence: float) -> str:
    # Build prompt: model should answer only from retrieved context.
    prompt = f"""You are a helpful assistant. Answer based ONLY on the provided context.
Be concise and accurate. If context doesn't contain information, say: "I don't have this information."

Context:
{context}

Question:
{question}

Answer:
"""

    # Ask chat model to generate final answer with streaming for faster response
    response = ollama_client.generate(
        model=CHAT_MODEL,
        prompt=prompt,
        stream=False,  # Collect full response
        options={"temperature": 0.1, "num_predict": 150},  # Limit tokens for faster response
    )

    # Return model text safely.
    answer = response.get("response", "").strip()
    return answer if answer else "I don't have this information."


def run_chat_loop() -> None:
    # Interactive chatbot loop - load articles ONCE at startup
    print("Loading knowledge base...")
    load_and_embed_articles_lazy()
    print("Chatbot is ready. Type 'exit' to stop.\n")

    while True:
        user_input = input("How may I assist you? ").strip()

        # Exit commands for clean stop.
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        # Ignore empty messages.
        if not user_input:
            print("Please type a question.")
            continue

        # Retrieve context for current user question with confidence scores
        # No need to reload - articles loaded once at startup
        context, confidence, chunk_count = retrieve_context(user_input, n_results=TOP_K)
        
        # If no context found, inform user
        if not context:
            print("\nAnswer:")
            print("❌ I don't have this information - no relevant content found.")
            print("-" * 60)
            continue
        
        # Show retrieval details
        print(f"\n📊 Retrieved {chunk_count} relevant chunk(s) | Confidence: {confidence:.0%}")
        
        # Generate answer from retrieved context + question.
        answer = generate_answer(user_input, context, confidence)

        print("\nAnswer:")
        print(f"✓ {answer}")
        
        # Confidence indicator
        if confidence >= 0.60:
            print("📈 Confidence: 🟢 HIGH")
        elif confidence >= 0.50:
            print("📈 Confidence: 🟡 MEDIUM")
        else:
            print("📈 Confidence: 🔴 LOW")
        print("-" * 60)


# ==============================
# PHASE 5: PROGRAM ENTRY POINT
# ==============================
if __name__ == "__main__":
    # Step 1: Start chatbot loop (articles loaded on-demand).
    run_chat_loop()
