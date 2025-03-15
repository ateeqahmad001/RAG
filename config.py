from langchain_groq import ChatGroq

def create_llm(groq_api_key: str, model: str = "gemma2-9b-it"):
    return ChatGroq(model=model, groq_api_key=groq_api_key)

groq_api_key = "dummy_api_key"
model = "gemma2-9b-it"
llm = create_llm(groq_api_key, model)

