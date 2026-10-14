import os 
from langgraph.graph import StateGraph , START, END
from typing import TypedDict , Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-120b")

def mearge_score_dic(existing :dict, new_dict :dict)->dict:
    if existing is None:
        return new_dict
    for key, value in new_dict.items():
        if key in existing:
            existing[key] += value
        else:
            existing[key] = value
    return existing

class Analyzer(TypedDict):
    row_input: str
    safety_scores: Annotated[dict[str, int ],mearge_score_dic]






#nodes 
def toxicity_node(state: Analyzer) -> dict:
    print("\n [Branch 1] Analyzing Toxicity and Hate Speech...")
    prompt = (
        "Analyze the following text for profanity, aggression, hate speech, or toxicity. "
        "Provide a score from 0 to 100, where 0 means perfectly clean and 100 means highly toxic. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['row_input']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under our single state key
    return {"safety_scores": {"toxicity_level": score}}

def copyright_node(state: Analyzer) -> dict:
    print("\n🔏 [Branch 2] Analyzing Copyright & Originality Risks...")
    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal, "
        "or presents a corporate trademark risk. Provide a score from 0 to 100, "
        "where 0 means entirely original and 100 means high risk. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['row_input']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under the EXACT SAME state key
    return {"safety_scores": {"copyright_risk": score}}


def culture_node(state: Analyzer) -> dict:
    print("\n🌍 [Branch 3] Analyzing Regional & Cultural Sensitivity...")
    prompt = (
        "Analyze the following text for regional sensitivities, political landmines, "
        "or cultural insensitivity that might offend a global audience. Provide a score from 0 to 100, "
        "where 0 means completely safe and 100 means highly offensive. "
        "Return ONLY the plain integer number, nothing else.\n\n"
        f"Text:\n{state['row_input']}"
    )
    response = llm.invoke(prompt)
    try:
        score = int(response.content.strip())
    except ValueError:
        score = 0
        
    # Return a sub-dictionary under the EXACT SAME state key
    return {"safety_scores": {"cultural_insensitivity": score}}




bul = StateGraph(Analyzer)
bul.add_node("toxicity_node", toxicity_node)
bul.add_node("copyright_node", copyright_node)
bul.add_node("culture_node", culture_node)


bul.add_edge(START, "toxicity_node")
bul.add_edge(START, "copyright_node")
bul.add_edge(START , "culture_node")


bul.add_edge("toxicity_node", END)
bul.add_edge("copyright_node", END)
bul.add_edge("culture_node", END)


app = bul.compile()

sample_script = """
    Yo guys! Welcome back to the stream. Today I am going to show you how to hack into 
    your friend's system using a script I copied directly from an online forum. 
    Honestly, traditional security protocols are absolute garbage and anyone still using 
    them is an absolute idiot. Let's dive into the code!
    """


initial_stat = {
    "row_input" : sample_script,
    "safety_scores": {}
}



final_stat=app.invoke(initial_stat)


print(final_stat["safety_scores"])


print(final_stat)