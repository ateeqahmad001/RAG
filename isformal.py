from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
import config
from state import graph_state

class isformal_prompt(BaseModel):
    score_: str = Field(description="Indicates the query type. 'no' for Factual-Related queries, 'yes' for formal conversational queries")

isformal_doc_llm = config.llm.with_structured_output(isformal_prompt)

isformal_system_message="""
Your task is to categorize the user's question into one of two categories:
**Formal/Chit-Chat Query**: These queries are conversational, informal, or involve casual communication. Examples include greetings, introductions, or simple questions about the user's interests, hobbies, or daily life (e.g., "Hi," "How are you?" "Tell me about yourself," etc.).
Response: "yes"

**Factual-Related Query**: These queries are specific, information-driven, and often related to finance, business, or technical topics such as investments, markets, inflation, taxes, or other well-defined subjects requiring detailed knowledge.
Response: "no"

### Rules for Categorization:

- If the query explicitly relates to finance, business, or factual topics(Factual-Related), respond with "no".
- If the query is conversational, chit-chat, or casual in tone, respond with "yes".
- In cases of ambiguity or overlap, default to "no".
"""
isformal_documents_prompt = "User's question: {question}"


def isformal_check(state:graph_state):
    ques = state["question"]
    prmpt = isformal_documents_prompt.format(question=ques)
    sc = isformal_doc_llm.invoke(
            [
                (SystemMessage(content=isformal_system_message)),
                (HumanMessage(content=prmpt))
            ]
      )
    grade_ = sc.score_
    if grade_=="no":
        return {"generation":[],"question":ques}
    else:
        return {"generation":sc.score_,"question":ques}
    
def decide_isformal(state:graph_state):
    gener = state["generation"]
    if not gener:
        return "Information related"
    else:
        return "formal"
    
formal_prompt = """you are a general assistant who handles formal questions or questions related to the user's lifestyle or general conversation.
Please handle it politely, professionally, and appropriately, providing clear and helpful responses tailored to the context of the question.
Question : {question}
"""

def formal_llm(state:graph_state):
    question = state["question"]
    pmt = formal_prompt.format(question=question)
    gener = config.llm.invoke([HumanMessage(content=pmt)])
    return {"generation":gener,"question":question}

