from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

import config
from state import GraphState

FORMALITY_SYSTEM_PROMPT = """
Your task is to categorize the user's question into one of two categories:

**Formal/Chit-Chat Query**: These queries are conversational, informal, or involve casual communication.
Examples include greetings, introductions, or simple questions about the user's interests, hobbies, or daily life
(e.g., "Hi," "How are you?" "Tell me about yourself," etc.).
Response: "yes"

**Factual-Related Query**: These queries are specific, information-driven, and often related to finance, business,
or technical topics such as investments, markets, inflation, taxes, or other well-defined subjects requiring
detailed knowledge.
Response: "no"

Rules for Categorization:
- If the query explicitly relates to finance, business, or factual topics, respond with "no".
- If the query is conversational, chit-chat, or casual in tone, respond with "yes".
- In cases of ambiguity or overlap, default to "no".
"""

FORMALITY_USER_PROMPT = "User's question: {question}"

FORMAL_RESPONSE_PROMPT = """You are a general assistant who handles formal questions or questions related
to the user's lifestyle or general conversation. Please handle it politely, professionally, and appropriately,
providing clear and helpful responses tailored to the context of the question.

Question: {question}
"""


class FormalityScore(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    score: str = Field(
        description="Indicates the query type. 'no' for factual queries, 'yes' for conversational queries."
    )


def check_formality(state: GraphState) -> dict:
    question = state["question"]
    visited = list(state.get("visited_nodes", []))
    visited.append("isformal_check")

    structured_llm = config.llm.with_structured_output(FormalityScore)
    user_prompt = FORMALITY_USER_PROMPT.format(question=question)

    result = structured_llm.invoke(
        [
            SystemMessage(content=FORMALITY_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    is_conversational = result.score == "yes"
    return {
        "generation": result.score if is_conversational else None,
        "question": question,
        "visited_nodes": visited,
    }


def route_after_formality_check(state: GraphState) -> str:
    if state.get("generation"):
        return "conversational"
    return "factual"


def handle_conversational_query(state: GraphState) -> dict:
    question = state["question"]
    visited = list(state.get("visited_nodes", []))
    visited.append("formal_responder")

    prompt = FORMAL_RESPONSE_PROMPT.format(question=question)
    response = config.llm.invoke([HumanMessage(content=prompt)])

    return {
        "generation": response.content,
        "question": question,
        "visited_nodes": visited,
    }
