from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from config import llm
from state import graph_state

class grade_doc(BaseModel):
    score : str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

grade_doc_llm = llm.with_structured_output(grade_doc)
grade_systm_msg = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_documents_prompt = "Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}"

def grade_document(state:graph_state):
    ques = state["question"]
    docs = state["documents"]

    filter_doc = []
    for d in docs:
        prmpt = grade_documents_prompt.format(document=d.page_content,question=ques)
        sc = grade_doc_llm.invoke(
            [
                (SystemMessage(content=grade_systm_msg)),
                (HumanMessage(content=prmpt))
            ]
        )
        grade = sc.score
        if grade == "yes":
            filter_doc.append(d)
        else:
            continue
    return {"documents":filter_doc,"question":ques}

def decide_grade(state: graph_state):
    filter_docs = state['documents']
    if not filter_docs:
        state.setdefault('attempts', 0)
        state['attempts'] += 1
        if state['attempts'] >= 3:
            print("Attempted responses:", state['generation'])
            return "trouble understanding"
        return "non relevant"
    else:
        state['non_relevant_attempts'] = 0
        return "some relevant"