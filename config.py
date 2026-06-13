from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")
DEFAULT_MODEL: str = os.environ.get("DEFAULT_MODEL", "llama-3.3-70b-versatile")

AVAILABLE_MODELS: list[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "meta-llama/llama-prompt-guard-2-22m",
]

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY


def build_llm(groq_api_key: str, model_name: str = DEFAULT_MODEL) -> ChatGroq:
    return ChatGroq(model=model_name, groq_api_key=groq_api_key)


llm: ChatGroq = build_llm(GROQ_API_KEY, DEFAULT_MODEL) if GROQ_API_KEY else None
