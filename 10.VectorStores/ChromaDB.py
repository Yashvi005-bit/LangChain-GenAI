from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document


embedding = HuggingFaceEmbeddings(
   model_name='sentence-transformers/all-MiniLM-L6-v2'
)

doc1 = Document(
    page_content="LangChain helps build LLM-powered applications.",
    metadata={"source": "langchain"}
)

doc2 = Document(
    page_content="ChromaDB is a vector database for storing embeddings.",
    metadata={"source": "chroma"}
)

doc3 = Document(
    page_content="FastAPI is commonly used for building Python APIs.",
    metadata={"source": "fastapi"}
)

docs = [doc1,doc2,doc3]

vector_store = Chroma(
    embedding_function= embedding,
    persist_directory="./10.VectorStores/chroma_db",
    collection_name='sample'

)
vector_store.add_documents(docs)

print("Document stored!")


#View Documents
view = vector_store.get(include=['embeddings','metadatas','documents'])
print(view)

#similarity check
search = vector_store.similarity_search(
    query='What is ChromaDB',
    k=2
)
print(search)