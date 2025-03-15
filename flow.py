from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

from guardrail import guardril_check,decide_guardril
from isformal import isformal_check,decide_isformal,formal_llm
from grade_document import grade_document,decide_grade
from hallucination import grade_halluc
from generate_response import generate_response
from web_search import web_search

from langgraph.graph import StateGraph,START,END
from IPython.display import Image, display
from config import llm

def run_pipeline(query: str,pdf_path: str) -> str:
    from state import graph_state
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    loader = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(loader)
    vector = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vector.as_retriever(search_kwargs={"k": 3})

    def retrieve_document(state: graph_state):
        question = state['question']
        document = retriever.invoke(question) 
        return {"documents": document, "question": question}

    question = {"question": query}

    flow = StateGraph(graph_state)
    flow.add_node("guardril_check",guardril_check)
    flow.add_node("isformal_check",isformal_check)
    flow.add_node("web_search",web_search)
    flow.add_node("retrieve",retrieve_document)
    flow.add_node("grade_document",grade_document)
    flow.add_node("generate_response",generate_response)
    flow.add_node("formal",formal_llm)
    flow.add_edge(START,"guardril_check")
    flow.add_conditional_edges(
        "guardril_check",
        decide_guardril,
        {
            "Inappropriate query":END,
            "fine":"isformal_check",
        }
    )
    flow.add_conditional_edges(
        "isformal_check",
        decide_isformal,
        {
            "Information related":"retrieve",
            "formal":"formal",
        }
    )
    flow.add_edge("retrieve","grade_document")
    flow.add_conditional_edges(
        "grade_document",
        decide_grade,
        {
            "non relevant":"web_search",
            "some relevant":"generate_response",
            "trouble understanding":"generate_response"
        }
    )
    flow.add_edge("web_search","generate_response")
    flow.add_conditional_edges(
        "generate_response",
        grade_halluc,
        {
            "supported":END,
            "not supported":"retrieve"
        }
    )
    flow.add_edge("formal",END)
    flow = flow.compile()
    response = flow.invoke(question,{"recursion_limit":10})
    return response['generation']