from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path = "8.DocumentLoader/Sample",
    glob = "*.pdf",
    loader_cls = PyPDFLoader
)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)

