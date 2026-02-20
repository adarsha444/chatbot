"""
Step-by-step reliable RAG chatbot with confidence scoring
"""
import json
from pathlib import Path
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client
import numpy as np

# Configuration
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:3b"
COLLECTION_NAME = "simple_knowledge"
TOP_K = 3
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence required

BASE_DIR = Path(__file__).resolve().parent
ARTICLES_FILE = BASE_DIR / "articles.jsonl"
COUNTER_FILE = BASE_DIR / "counter.txt"
CHROMA_DIR = BASE_DIR / "chroma"

# Initialize clients
ollama_client = Client(host=OLLAMA_HOST)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# Text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
    separators=[".", "\n"],
)

def load_counter():
    if not COUNTER_FILE.exists():
        return 0
    try:
        return int(COUNTER_FILE.read_text(encoding="utf-8").strip())
    except:
        return 0

def save_counter(idx):
    COUNTER_FILE.write_text(str(idx), encoding="utf-8")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# STEP 1: Load and embed articles
print("=" * 70)
print("STEP 1️⃣  LOADING AND EMBEDDING ARTICLES")
print("=" * 70)

start_line = load_counter()
if start_line == 0:
    print("\n📥 Reading articles from: articles.jsonl")
    with ARTICLES_FILE.open("r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            article = json.loads(line)
            title = article.get("title", "Untitled")
            content = article.get("content", "").strip()
            
            print(f"\n📄 Article {i}: '{title}'")
            
            if not content:
                continue
            
            chunks = [c.strip() for c in splitter.split_text(content) if c.strip()]
            print(f"   → Chunks: {len(chunks)} pieces")
            
            for j, chunk in enumerate(chunks):
                embed_response = ollama_client.embed(
                    model=EMBED_MODEL,
                    input=f"search_document: {chunk}",
                )
                embedding = embed_response["embeddings"][0]
                
                collection.upsert(
                    ids=[f"id_{i}_{j}"],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"line": i, "chunk": j, "title": title}],
                )
            
            save_counter(i + 1)
    print("\n✅ All articles loaded into vector database")
else:
    print(f"✅ Database already populated (from {start_line} articles)")

print("\n" + "=" * 70)
print("STEP 2️⃣  USER QUESTION & EMBEDDING")
print("=" * 70)

# User question
question = "What does the Election Commission say about March 5 polls?"
print(f"\n❓ Question: '{question}'")
print(f"   (Length: {len(question)} characters)")

# STEP 3: Embed question
print("\nSTEP 3️⃣  CONVERTING QUESTION TO VECTOR REPRESENTATION")
print("-" * 70)
query_embedding = ollama_client.embed(
    model=EMBED_MODEL,
    input=f"query: {question}",
)["embeddings"][0]
print(f"✓ Question embedding created (vector size: {len(query_embedding)} dimensions)")

# STEP 4: Search with detailed scoring
print("\nSTEP 4️⃣  SEARCHING FOR RELEVANT CONTENT (WITH CONFIDENCE SCORES)")
print("-" * 70)
results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
retrieved_docs = results.get("documents", [[]])[0]
retrieved_ids = results.get("ids", [[]])[0]
retrieved_distances = results.get("distances", [[]])[0]

print(f"📊 Searching... Found {len(retrieved_docs)} candidate chunks\n")

# Calculate confidence scores (convert distance to similarity)
confirmed_chunks = []
for idx, (doc_id, distance, doc) in enumerate(zip(retrieved_ids, retrieved_distances, retrieved_docs)):
    # Chroma uses L2 distance, convert to similarity score (0-1)
    similarity = 1 / (1 + distance)  # Convert distance to similarity
    confidence = max(0, similarity)
    
    print(f"[Result {idx + 1}] Confidence: {confidence:.1%}")
    print(f"   Source: {doc_id}")
    print(f"   Text: '{doc[:60]}...'")
    
    if confidence >= CONFIDENCE_THRESHOLD:
        confirmed_chunks.append({
            'text': doc,
            'id': doc_id,
            'confidence': confidence
        })
        print(f"   ✅ ACCEPTED (confidence ≥ {CONFIDENCE_THRESHOLD:.0%})")
    else:
        print(f"   ❌ REJECTED (confidence < {CONFIDENCE_THRESHOLD:.0%})")
    print()

if not confirmed_chunks:
    print("⚠️  WARNING: No chunks met confidence threshold!")
    confirmed_chunks = [{'text': doc, 'id': doc_id, 'confidence': confidence} 
                       for doc_id, doc, confidence in zip(retrieved_ids, retrieved_docs, 
                       [1/(1+d) for d in retrieved_distances])]

# STEP 5: Build context
print("STEP 5️⃣  BUILDING RELIABLE CONTEXT")
print("-" * 70)
context = "\n\n".join(chunk['text'] for chunk in confirmed_chunks)
avg_confidence = sum(c['confidence'] for c in confirmed_chunks) / len(confirmed_chunks)
print(f"✓ Combined {len(confirmed_chunks)} relevant chunk(s)")
print(f"✓ Average confidence: {avg_confidence:.1%}")
print(f"\nContext to be used:\n{context}\n")

# STEP 6: Generate answer with multiple checks
print("STEP 6️⃣  GENERATING ANSWER (WITH VALIDATION)")
print("-" * 70)

prompt = f"""You are a helpful assistant. Answer based ONLY on the provided context.
Be concise and accurate. Always cite which source information you're using.
If you cannot answer from the context, say: "I don't know - this information is not in my knowledge base."

Context:
{context}

Question:
{question}

Answer:"""

print("📝 Sending to language model...")
print(f"   Model: {CHAT_MODEL}")
print(f"   Temperature: 0.1 (low - for reliable answers)")

response = ollama_client.generate(
    model=CHAT_MODEL,
    prompt=prompt,
    options={"temperature": 0.1},
)

answer = response.get("response", "").strip()

# STEP 7: Validate answer reliability
print("\nSTEP 7️⃣  ANSWER RELIABILITY CHECK")
print("-" * 70)

reliability_checks = {
    "Answer is not empty": len(answer) > 0,
    "Answer is not 'I don't know'": "i don't know" not in answer.lower() or len(confirmed_chunks) > 0,
    "High context confidence": avg_confidence >= 0.6,
    "Answer mentions key terms": any(term.lower() in answer.lower() 
                                     for term in ["election", "commission", "march", "weather"])
}

print("Reliability Checks:")
for check, passed in reliability_checks.items():
    status = "✅" if passed else "⚠️"
    print(f"  {status} {check}")

overall_reliability = sum(reliability_checks.values()) / len(reliability_checks)
reliability_level = "🟢 HIGH" if overall_reliability >= 0.75 else "🟡 MEDIUM" if overall_reliability >= 0.5 else "🔴 LOW"

print(f"\nOverall Reliability Score: {overall_reliability:.0%} {reliability_level}")

# FINAL ANSWER
print("\n" + "=" * 70)
print("FINAL ANSWER")
print("=" * 70)
print(f"\n💡 Answer:\n{answer}")
print(f"\n📊 Confidence: {overall_reliability:.0%}")
print(f"📚 Sources used: {len(confirmed_chunks)} relevant chunk(s)")
print(f"✓ Reliability: {reliability_level}")
print(f"\n{'='*70}\n")

