from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel , RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

llm1 = HuggingFaceEndpoint(
     repo_id= "meta-llama/Llama-3.1-8B-Instruct",
     task="text-generation" 
)

model = ChatHuggingFace(llm=llm1)
class Feedback(BaseModel):
    sentiment : Literal['positive','negative'] = Field(description= 'Give the sentiment of the feedback')

parser = StrOutputParser()
# We are using pydantic output parser because we want the result to me consistent as our future results will be based on the feedback sentiment provided by LLM
parser2 = PydanticOutputParser(pydantic_object = Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables = ['feedback'],
    partial_variables= {'format_instruction' : parser2.get_format_instructions()}
)
classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = 'Write an appropiate response for positive feedback \n {feedback}',
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = 'Write an appropiate response for negative feedback \n {feedback}',
    input_variables = ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser ),
     RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain
print(chain.invoke({'feedback' : 'This is a terrible phone'}))







