from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

# breakpoint thershold type is set to standard deviation which means the standard deviation of the similarity and the breakpoint thershold amount is the breaking point of two similarities.
text_splitter = SemanticChunker(
    embedding,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

sample = """
Artificial Intelligence is transforming industries.
Machine learning is a branch of AI.
Deep learning uses neural networks.

Football is one of the most popular sports.
The FIFA World Cup is watched worldwide.

Python is a programming language.
It is widely used for AI and web development.
"""

docs = text_splitter.create_documents([sample])

print(len(docs))
print(docs)