from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph

import config
from generate_response import generate_response
from grade_document import grade_documents, increment_attempts, route_after_grading
from guardrail import route_after_guardrail, run_guardrail
from hallucination import route_on_hallucination_check
from isformal import (
    check_formality,
    handle_conversational_query,
    route_after_formality_check,
)
from state import GraphState
from web_search import perform_web_search, return_fallback_response

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_K = 3
MAX_RECURSION_LIMIT = 15


@st.cache_resource
def build_embeddings_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def build_faiss_index_from_pdfs(pdf_file_bytes_list: tuple[tuple[str, bytes], ...], _session_id: str) -> FAISS:
    embeddings = build_embeddings_model()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    all_documents = []

    for pdf_name, pdf_bytes in pdf_file_bytes_list:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_bytes)
            temp_pdf_path = temp_file.name

        try:
            raw_documents = PyPDFLoader(temp_pdf_path).load()
            for doc in raw_documents:
                doc.metadata["pdf_name"] = pdf_name
            split_documents = text_splitter.split_documents(raw_documents)
            all_documents.extend(split_documents)
        finally:
            os.unlink(temp_pdf_path)

    return FAISS.from_documents(all_documents, embedding=embeddings)

@st.cache_resource
def build_compiled_pipeline(_faiss_index: FAISS, _session_id: str) -> StateGraph:
    retriever = _faiss_index.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    def retrieve_documents(state: GraphState) -> dict:
        question = state["question"]
        visited = list(state.get("visited_nodes", []))
        visited.append("retrieve")
        retrieved = retriever.invoke(question)
        return {"documents": retrieved, "question": question, "visited_nodes": visited}

    pipeline = StateGraph(GraphState)

    pipeline.add_node("guardrail", run_guardrail)
    pipeline.add_node("isformal_check", check_formality)
    pipeline.add_node("formal_responder", handle_conversational_query)
    pipeline.add_node("retrieve", retrieve_documents)
    pipeline.add_node("grade_document", grade_documents)
    pipeline.add_node("increment_attempts", increment_attempts)
    pipeline.add_node("web_search", perform_web_search)
    pipeline.add_node("generate_response", generate_response)
    pipeline.add_node("fallback", return_fallback_response)

    pipeline.add_edge(START, "guardrail")

    pipeline.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "blocked": END,
            "allowed": "isformal_check",
        },
    )

    pipeline.add_conditional_edges(
        "isformal_check",
        route_after_formality_check,
        {
            "conversational": "formal_responder",
            "factual": "retrieve",
        },
    )

    pipeline.add_edge("formal_responder", END)
    pipeline.add_edge("retrieve", "grade_document")

    pipeline.add_conditional_edges(
        "grade_document",
        route_after_grading,
        {
            "relevant_docs_found": "generate_response",
            "no_relevant_docs": "increment_attempts",
            "max_attempts_reached": "fallback",
        },
    )

    pipeline.add_edge("increment_attempts", "web_search")
    pipeline.add_edge("web_search", "grade_document")

    pipeline.add_conditional_edges(
        "generate_response",
        route_on_hallucination_check,
        {
            "grounded": END,
            "hallucinated": "retrieve",
        },
    )

    pipeline.add_edge("fallback", END)

    return pipeline.compile()

def run_pipeline(
    query: str,
    pdf_file_bytes_list: tuple[tuple[str, bytes], ...],
    groq_api_key: str,
    model_name: str,
    conversation_context: str = "",
) -> dict:
    config.llm = config.build_llm(groq_api_key, model_name)

    from streamlit.runtime.scriptrunner import get_script_run_ctx
    session_id = get_script_run_ctx().session_id
    faiss_index = build_faiss_index_from_pdfs(pdf_file_bytes_list,session_id)
    compiled_pipeline = build_compiled_pipeline(faiss_index,session_id)

    initial_state: GraphState = {
        "question": query,
        "generation": None,
        "documents": [],
        "attempts": 0,
        "web_searched": False,
        "visited_nodes": [],
        "conversation_context": conversation_context,
    }

    final_state = compiled_pipeline.invoke(
        initial_state, {"recursion_limit": MAX_RECURSION_LIMIT}
    )

    return final_state
