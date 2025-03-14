from typing import TypedDict, List
from langchain.docstore.document import Document

class graph_state(TypedDict):
    question: str
    generation: str
    documents: List[Document]
    attempts: int
