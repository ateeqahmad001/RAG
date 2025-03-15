from langchain_core.messages import HumanMessage
import config
from state import graph_state

rag_prompt = """
Please read the following question and its accompanying context carefully. Based on the provided information, construct a detailed and well-organized answer using proper Markdown formatting. Your response should include:

**Clear Headings**: Use H1, H2, or H3 tags where appropriate to segment your answer.
**Bullet Points/Numbered Lists**: Include lists to highlight key points or steps.
**Concise Explanations**: Ensure that each section is clearly explained and directly addresses the question and context.
**Summary or Conclusion**: Wrap up your answer with a brief summary if necessary.

Question: {question} 

Context: {context} 

Answer:"""


def generate_response(state:graph_state):
    question = state["question"]
    doc = state["documents"]
    formatted_doc = "\n\n".join(
    d.page_content if hasattr(d, "page_content") else str(d) for d in doc
    )
    # formatted_doc = "\n\n".join(d.page_content for d in doc)

    prompt = rag_prompt.format(context=formatted_doc,question=question)
    generate = config.llm.invoke([HumanMessage(content=prompt)])
    return {"documents":doc,"question":question,"generation":generate}