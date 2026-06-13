from __future__ import annotations

from langchain_core.messages import HumanMessage

import config
from state import GraphState

RAG_PROMPT = """Please read the following question and its accompanying context carefully. Based on the provided
information, construct a detailed and well-organized answer using proper Markdown formatting. Your response should include:

**Clear Headings**: Use H1, H2, or H3 tags where appropriate to segment your answer.
**Bullet Points/Numbered Lists**: Include lists to highlight key points or steps.
**Concise Explanations**: Ensure that each section is clearly explained and directly addresses the question and context.
**Summary or Conclusion**: Wrap up your answer with a brief summary if necessary.

Question: {question}

Context: {context}

Answer:"""


def generate_response(state: GraphState) -> dict:
    question = state["question"]
    documents = state["documents"]
    visited = list(state.get("visited_nodes", []))
    visited.append("generate_response")

    formatted_context = "\n\n".join(
        d.page_content if hasattr(d, "page_content") else str(d) for d in documents
    )

    conversation_context = state.get("conversation_context", "")
    full_question = (
        f"{conversation_context}\n\nCurrent question: {question}"
        if conversation_context
        else question
    )

    prompt = RAG_PROMPT.format(context=formatted_context, question=full_question)
    response = config.llm.invoke([HumanMessage(content=prompt)])

    return {
        "documents": documents,
        "question": question,
        "generation": response.content,
        "visited_nodes": visited,
    }
