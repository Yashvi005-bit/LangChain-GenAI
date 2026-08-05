from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

embedding = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

documents =[
Document(page_content="Langchain make it easy to work with LLMs"),
Document(page_content="Langchain is used to build LLM based appliaction"),
Document(page_content="Embeddings are vector representation of text"),
Document(page_content="MMR helps to get diverse result when doing similarity search")
]

vector_store = FAISS.from_documents(
    embedding= embedding,
    documents=documents,
   
)

retriver = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3, "lambda_mult":1}
)

query = "What is langchain?"

docs = retriver.invoke(query)
for doc in docs:
    print(doc) 