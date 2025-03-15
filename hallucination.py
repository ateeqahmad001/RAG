from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from main import llm
from state import graph_state

class hallucinate(BaseModel):
    score_ : str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

grade_halluc_llm = llm.with_structured_output(hallucinate)
grade_halluc_sys_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
grade_halluc_prompt = "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"

def grade_halluc(state:graph_state):
    gener = state["generation"]
    docs = state["documents"]
    format_doc = "\n\n".join(
    d.page_content if hasattr(d, "page_content") else str(d) for d in docs
    )
    halluc_prompt = grade_halluc_prompt.format(documents=format_doc,generation=gener)
    sc = grade_halluc_llm.invoke(
        [
            (SystemMessage(content=grade_halluc_sys_prompt)),
            (HumanMessage(content=halluc_prompt))
        ]
    )
    grade = sc.score_
    if grade == "yes":
        return "supported"
    else:
        return "not supported"
    