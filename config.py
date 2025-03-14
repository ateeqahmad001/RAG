# config.py
from langchain.llms import ChatGroq

def create_llm(groq_api_key: str, model: str = "gemma2-9b-it"):
    return ChatGroq(model=model, groq_api_key=groq_api_key)

groq_api_key = "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"
model = "gemma2-9b-it"
llm = create_llm(groq_api_key, model)

