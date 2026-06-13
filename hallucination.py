from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

import config
from state import GraphState

HALLUCINATION_GRADER_SYSTEM_PROMPT = """You are a grader assessing whether an LLM generation is grounded in
and supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in and supported by the set of facts."""

HALLUCINATION_GRADER_USER_PROMPT = (
    "Set of facts:\n\n{documents}\n\nLLM generation: {generation}"
)


class HallucinationScore(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    score: str = Field(
        description="Whether the generation is grounded in the facts. 'yes' means grounded, 'no' means hallucinated."
    )


def route_on_hallucination_check(state: GraphState) -> str:
    generation = state["generation"]
    documents = state["documents"]

    formatted_documents = "\n\n".join(
        d.page_content if hasattr(d, "page_content") else str(d) for d in documents
    )

    structured_llm = config.llm.with_structured_output(HallucinationScore)
    user_prompt = HALLUCINATION_GRADER_USER_PROMPT.format(
        documents=formatted_documents, generation=generation
    )

    result = structured_llm.invoke(
        [
            SystemMessage(content=HALLUCINATION_GRADER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    if result.score == "yes":
        return "grounded"
    return "hallucinated"
