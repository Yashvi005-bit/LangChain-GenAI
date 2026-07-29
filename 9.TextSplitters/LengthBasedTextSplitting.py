from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("9.TextSplitters/Sample_Document_for_PyPDFLoader.pdf")

docs = loader.load()


splitter = CharacterTextSplitter(
    chunk_size= 100,
    chunk_overlap = 0,
    separator = ''
)

result = splitter.split_documents(docs)
print(result[1].page_content)
