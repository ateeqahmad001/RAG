from __future__ import annotations

from dataclasses import field
from typing import Annotated, Optional

from langchain.docstore.document import Document
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class GraphState(TypedDict):
    question: str
    generation: Optional[str]
    documents: list[Document]
    attempts: int
    web_searched: bool
    visited_nodes: list[str]
