from langchain_community.retrievers import WikipediaRetriever

retriver = WikipediaRetriever(
    top_k_results=2,
    lang='en'
)

query= "GenerativeAI is in growing demand"

docs = retriver.invoke(query)

for doc in docs:
    print(doc.page_content)
