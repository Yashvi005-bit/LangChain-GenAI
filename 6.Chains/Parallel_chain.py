from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm1 = HuggingFaceEndpoint(
     repo_id= "meta-llama/Llama-3.1-8B-Instruct",
     task="text-generation" 
)
# We can also use a different LLM Model
llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short question answers from the following text \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template = 'Merge the provided notes and quiz into single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain



text = """
Generative artificial intelligence (GenAI) is a subfield of artificial intelligence (AI) that uses generative models to generate text, images, videos, audio, software code (vibe coding) or other forms of data.[1] These models learn the underlying patterns and structures of their training data, and use them to generate new data[2] in response to input, which often takes the form of natural language prompts.

The prevalence of generative AI tools has increased significantly since the AI boom in the 2020s. This boom was made possible by improvements in deep neural networks, particularly large language models (LLMs), which are based on the transformer architecture. Generative AI applications include chatbots such as ChatGPT, Claude, Copilot, DeepSeek, Doubao, Google Gemini, Grok and Qwen; text-to-image models such as DALL-E, Firefly, Stable Diffusion, and Midjourney; and text-to-video models such as Veo, LTX and Sora.

Companies in a variety of sectors have used generative AI, including those in software development, healthcare,[8] finance,[9] entertainment,customer service, sales and marketing,art, writing,and product design.

Generative AI has been used for cybercrime, and to deceive and manipulate people through fake news and deepfakes. Generative AI models have been trained on copyrighted works without the rightholders' permission.Many generative AI systems use large-scale data centers, whose environmental impacts include electronic waste, consumption of fresh water for cooling, and high energy consumption that is estimated to be growing steadily.
"""

result = chain.invoke({'text' : text})
print(result)

chain.get_graph().print_ascii()

