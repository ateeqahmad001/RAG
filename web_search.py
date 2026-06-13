from __future__ import annotations

from langchain.docstore.document import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from state import GraphState

tavily_search_tool = TavilySearchResults(k=3)

FALLBACK_RESPONSE = (
    "I was unable to find relevant information to answer your question. "
    "Please try rephrasing your query or uploading a more relevant document."
)


def perform_web_search(state: GraphState) -> dict:
    question = state["question"]
    existing_documents = list(state.get("documents", []))
    visited = list(state.get("visited_nodes", []))
    visited.append("web_search")

    search_results = tavily_search_tool.invoke({"query": question})

    web_documents = []
    for result in search_results:
        web_doc = Document(
            page_content=result.get("content", str(result)),
            metadata={
                "source": result.get("url", "web"),
                "title": result.get("title", "Web Result"),
                "pdf_name": "Web Search",
            },
        )
        web_documents.append(web_doc)

    combined_documents = existing_documents + web_documents

    return {
        "documents": combined_documents,
        "question": question,
        "web_searched": True,
        "visited_nodes": visited,
    }


def return_fallback_response(state: GraphState) -> dict:
    visited = list(state.get("visited_nodes", []))
    visited.append("fallback")
    return {
        "generation": FALLBACK_RESPONSE,
        "visited_nodes": visited,
    }
