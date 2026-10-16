import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

INDEX_DIR = "faiss_indexes"


def get_retriever(pdf_path: str):
    """Loads existing FAISS embeddings from disk, or creates and saves them on first run."""
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    save_path = os.path.join(INDEX_DIR, f"{base_name}_faiss")

    # 1. Fast path: Load existing embeddings from disk if already created
    if os.path.exists(save_path):
        print(f"⚡ Loading cached embeddings for '{pdf_path}'...")
        db = FAISS.load_local(
            save_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    # 2. First-run path: Read PDF, create embeddings, and save to disk
    else:
        print(f"📄 Creating and saving embeddings for '{pdf_path}' (first-time setup)...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)
        db = FAISS.from_documents(docs, embeddings)

        os.makedirs(INDEX_DIR, exist_ok=True)
        db.save_local(save_path)
        print(f"✅ Saved embeddings to '{save_path}'")

    return db.as_retriever(search_kwargs={"k": 4})


academics_handbook = get_retriever("academics_handbook.pdf")
fee_structure = get_retriever("fee_structure.pdf")


# Fixed: Standardized all State keys so every node uses the exact same names
class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


def classifier_node(state: State) -> dict:
    """Look at the latest user message and decide which path to take."""
    last_message = state["messages"][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"

    return {"query_type": category}


def academic_rag_node(state: State) -> dict:
    """Retrieve context from the academics handbook."""
    query = state["messages"][-1].content
    docs = academics_handbook.invoke(query)
    content = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": content}


def fee_rag_node(state: State) -> dict:
    """Retrieve context from the fee structure handbook."""
    query = state["messages"][-1].content
    docs = fee_structure.invoke(query)
    content = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": content}


def general_node(state: State) -> dict:
    """Mark context as NO_RETRIEVAL_NEEDED for general queries."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


# Fixed: Changed `sate` -> `state` and `State[...]` -> `state[...]`
def route(state: State) -> str:
    if state["query_type"] == "academic":
        return "academic_rag_node"
    elif state["query_type"] == "fee":
        return "fee_rag_node"
    else:
        return "general_node"


# Build the graph
graph = StateGraph(State)

graph.add_node("classifier", classifier_node)
graph.add_node("academic_rag_node", academic_rag_node)
graph.add_node("fee_rag_node", fee_rag_node)
graph.add_node("general_node", general_node)
graph.add_node("response_node", response_node)

graph.add_edge(START, "classifier")
graph.add_conditional_edges("classifier", route)
graph.add_edge("academic_rag_node", "response_node")
graph.add_edge("fee_rag_node", "response_node")
graph.add_edge("general_node", "response_node")
graph.add_edge("response_node", END)

app = graph.compile()


# CLI Loop
print("\nWelcome to the College Assistant!\n")
print("Which programme are you in?")
print("1. BCA")
print("2. BBA")
print("3. B.Com (H)")

choice = input("\nEnter 1, 2 or 3: ").strip()

programme_map = {
    "1": "BCA",
    "2": "BBA",
    "3": "B.Com (H)",
}
student_programme = programme_map.get(choice, "BCA")

print(f"\nGreat! You're set as a {student_programme} student. (Type 'exit' or 'quit' to stop)\n")

while True:
    user_query = input("You: ").strip()

    if not user_query:
        continue
    if user_query.lower() in ["exit", "quit"]:
        break

    result = app.invoke(
        {
            "programme": student_programme,
            "messages": [("human", user_query)],
        }
    )

    print(f"\nAssistant: {result['messages'][-1].content}\n")