from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
# Introduction
This is the introduction section.

## What is AI?
Artificial Intelligence is the simulation of human intelligence.

## Applications
AI is used in healthcare, finance, and robotics.

# Conclusion
AI will continue to transform industries."""

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.MARKDOWN,
    chunk_size= 200,
    chunk_overlap = 0
)

chunks = splitter.split_text(text)
print(chunks)
print(len(chunks))