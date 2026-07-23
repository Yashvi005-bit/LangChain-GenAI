from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
     repo_id= "meta-llama/Llama-3.1-8B-Instruct",
     task="text-generation" 
)

model = ChatHuggingFace(llm = llm)

prompt = PromptTemplate(
    template = 'Write a summary for the following text \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

loader = TextLoader("8.DocumentLoader/AI.txt", encoding= 'utf-8')

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt | model | parser
print(chain.invoke({'text' : docs[0].page_content}))