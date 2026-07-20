from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
     repo_id= "meta-llama/Llama-3.1-8B-Instruct",
     task="text-generation" 
)
    
model = ChatHuggingFace(llm = llm)

#Template-1
template1 = PromptTemplate(
    template = 'Write a detailed review about {topic}',
    input_variable = ['topic']
)

#Template -2
template2 = PromptTemplate(
    template = 'Write a 5 line summary on the given text. /n {text}',
    input_variable= ['text']

)

prompt1 = template1.invoke({'topic' : 'Black hole'})

result = model.invoke(prompt1)

prompt2 = template2.invoke({'text' : result.content})

result1 = model.invoke(prompt2)

print(result1.content)
