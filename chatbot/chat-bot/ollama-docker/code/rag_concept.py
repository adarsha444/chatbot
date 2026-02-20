from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.globals import set_debug

set_debug(True)

# 1. Plain String Data - Demonstrating Context Loss
# The key information is split across chunks, making it incomplete
raw_text = """
John Smith was born in New York in 1985. He studied computer science at MIT from 2003 to 2007.
After graduation, he worked at Google for 5 years as a software engineer.

In 2012, John moved to San Francisco and started his own company called TechVision.
The company focused on artificial intelligence and machine learning solutions.

TechVision grew rapidly and by 2015 had over 100 employees. The company's main product
was an AI-powered customer service chatbot that could handle complex queries.

In 2018, John decided to sell TechVision to Amazon for $500 million. After the acquisition,
he became the VP of AI Research at Amazon, where he currently leads a team of 50 researchers.

John is married to Sarah Chen, whom he met at MIT. They have two children, Emma and Lucas.
The family lives in Seattle, Washington, where John commutes to Amazon's headquarters daily.
"""

# 2. Setup
# Lower temperature for more factual responses, higher for creative ones
llm = ChatOllama(model="qwen3:8b-q4_K_M", temperature=0.1)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# # 3. Process - Using small chunk size to force context splitting
# Small chunk size will split related information across chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)

initial_doc = Document(page_content=raw_text)
chunks = text_splitter.split_documents([initial_doc])

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

template = """
You are a helpful assistant. Answer the question based on the context provided. Use the information in the context to form your answer. If context does not have enough information just say "I don't know"
Context: {context}
Question: {question}
"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

# 7. Create the RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),  # Only retrieve 2 chunks
    chain_type_kwargs={"prompt": PROMPT},
)

# 5. Run - Query that requires information from multiple chunks
# This question needs info from chunks about his company AND the acquisition
query = "What company did John Smith start and who did he sell it to for how much?"

result = qa_chain.invoke(query)
print(result["result"])

