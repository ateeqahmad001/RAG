from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

import config
from state import GraphState

GRADER_SYSTEM_PROMPT = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

GRADER_USER_PROMPT = (
    "Here is the retrieved document:\n\n{document}\n\nHere is the user question:\n\n{question}"
)


class DocumentRelevanceScore(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    score: str = Field(
        description="Document relevance to the question. 'yes' means relevant, 'no' means not relevant."
    )


def grade_documents(state: GraphState) -> dict:
    question = state["question"]
    documents = state["documents"]
    visited = list(state.get("visited_nodes", []))
    visited.append("grade_document")

    structured_llm = config.llm.with_structured_output(DocumentRelevanceScore)
    relevant_documents = []

    for document in documents:
        page_content = (
            document.page_content if hasattr(document, "page_content") else str(document)
        )
        user_prompt = GRADER_USER_PROMPT.format(
            document=page_content, question=question
        )
        result = structured_llm.invoke(
            [
                SystemMessage(content=GRADER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        if result.score == "yes":
            relevant_documents.append(document)

    return {
        "documents": relevant_documents,
        "question": question,
        "visited_nodes": visited,
    }


def increment_attempts(state: GraphState) -> dict:
    return {"attempts": state.get("attempts", 0) + 1}


def route_after_grading(state: GraphState) -> str:
    relevant_documents = state["documents"]
    attempts = state.get("attempts", 0)

    if not relevant_documents:
        if attempts >= 3:
            return "max_attempts_reached"
        web_already_searched = state.get("web_searched", False)
        if web_already_searched:
            return "max_attempts_reached"
        return "no_relevant_docs"
    return "relevant_docs_found"
