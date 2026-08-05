from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma.vectorstores import Chroma
from langchain_core.documents import Document

embedding = HuggingFaceEmbeddings(
      model_name='sentence-transformers/all-MiniLM-L6-v2'
)

documents=[
    Document(page_content= "React is a JavaScript library."),
    Document(page_content="Langchain help devlopers to build LLM applications easily"),
    Document(page_content="OpenAI provides powerful embedding model"),
    Document(page_content="FastAPI is a Python framework.")
]

vector_stores = Chroma.from_documents(
    embedding=embedding,
    documents=documents,
    collection_name="My_collection",
     persist_directory="./11.Retrivers/chroma_db",
)

retriver = vector_stores.as_retriever(
    search_kwargs={"k":2}
)

query="What is Langchain?"
docs = retriver.invoke(query)
for doc in docs:
    print(doc)

    