from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#Chat Template 
Chat_template = ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name= "chat_history"), #This will store all the previous messages
    ('human', '{query}')
])

# Load chat history 
chat_history=[]
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())\
    
print(chat_history)

#create prompt
prompt = Chat_template.invoke({'chat_history':chat_history, 'query': 'Where is my refund?'})
print(prompt)