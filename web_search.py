from langchain_community.tools.tavily_search import TavilySearchResults
from state import graph_state

web_tool = TavilySearchResults(k=3)
def web_search(state:graph_state):
    ques = state['question']
    docs = state.get("documents",[])
    web_doc = web_tool.invoke({"query":ques})
    web_result = "\n".join([str(d) for d in web_doc])
    docs.append(web_result)
    return {"documents":docs,"question":ques}