import os
from typing import TypedDict, Annotated
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -------------------------------------------------------------------
# 1. Page Configuration & Styling
# -------------------------------------------------------------------
st.set_page_config(
    page_title="College AI Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

load_dotenv()

# -------------------------------------------------------------------
# 2. Cached Resources (LLM, Embeddings, Vector Stores, & Graph)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AI models and indexing college handbooks...")
def init_rag_graph():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    def build_retriever(pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Required PDF file not found: '{pdf_path}'")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)
        db = FAISS.from_documents(docs, embeddings)
        return db.as_retriever(search_kwargs={"k": 4})

    academics_handbook = build_retriever("academics_handbook.pdf")
    fee_structure = build_retriever("fee_structure.pdf")

    # Unified State definition (keys aligned across all nodes)
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
        """Retrieve relevant chunks from the Academics Handbook."""
        query = state["messages"][-1].content
        docs = academics_handbook.invoke(query)
        content = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": content}

    def fee_rag_node(state: State) -> dict:
        """Retrieve relevant chunks from the Fee Structure document."""
        query = state["messages"][-1].content
        docs = fee_structure.invoke(query)
        content = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": content}

    def general_node(state: State) -> dict:
        """Bypass vector retrieval for general conversation."""
        return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}

    def response_node(state: State) -> dict:
        """Generates the final answer, personalized using the student's programme."""
        query = state["messages"][-1].content
        programme = state.get("programme", "Unknown")
        context = state.get("retrieved_context", "NO_RETRIEVAL_NEEDED")

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

    def route(state: State) -> str:
        if state["query_type"] == "academic":
            return "academic_rag_node"
        elif state["query_type"] == "fee":
            return "fee_rag_node"
        else:
            return "general_node"

    # Build the LangGraph workflow
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

    return graph.compile()


# -------------------------------------------------------------------
# 3. Sidebar UI (Student Profile & Controls)
# -------------------------------------------------------------------
with st.sidebar:
    st.title("🎓 Student Profile")
    st.markdown("Select your enrolled programme to get personalized answers.")

    student_programme = st.selectbox(
        "Programme",
        options=["BCA", "BBA", "B.Com (H)"],
        index=0,
    )

    st.divider()
    st.subheader("📄 Knowledge Base Status")
    for pdf_file, label in [
        ("academics_handbook.pdf", "Academics Handbook"),
        ("fee_structure.pdf", "Fee Structure"),
    ]:
        if os.path.exists(pdf_file):
            st.success(f"✅ {label} loaded")
        else:
            st.error(f"❌ Missing `{pdf_file}`")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.caption("Powered by LangGraph, FAISS & Groq")

# -------------------------------------------------------------------
# 4. Main Chat Interface
# -------------------------------------------------------------------
st.title("🏛️ College Helpdesk Assistant")
st.caption(
    f"Currently assisting as a **{student_programme}** advisor. "
    "Ask about attendance, grading, exams, tuition fees, scholarships, or general queries!"
)

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": f"Hello! I'm your College AI Assistant. I see you're in the **{student_programme}** programme. How can I help you today?",
            "category": "general",
            "context": None,
        }
    ]

# Load the compiled LangGraph app safely
try:
    app = init_rag_graph()
except Exception as e:
    st.error(f"**Initialization Error:** {e}")
    st.info(
        "Ensure `academics_handbook.pdf` and `fee_structure.pdf` are in the same directory as `app.py` and your `GROQ_API_KEY` is set in `.env`."
    )
    st.stop()

# Badge styling map for query categories
CATEGORY_BADGES = {
    "academic": "📘 Academic Query",
    "fee": "💳 Fee Query",
    "general": "💬 General Chat",
}

# Render existing chat messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("category"):
            st.caption(CATEGORY_BADGES.get(msg["category"], "💬 General Chat"))
        st.markdown(msg["content"])
        if msg.get("context") and msg["context"] != "NO_RETRIEVAL_NEEDED":
            with st.expander("🔍 View Retrieved Document Context"):
                st.text(msg["context"])

# Quick-action starter prompts if no user messages yet
if len(st.session_state.chat_history) == 1:
    st.markdown("**Suggested questions:**")
    col1, col2, col3 = st.columns(3)
    sample_query = None
    if col1.button("📅 Attendance rules", use_container_width=True):
        sample_query = "What is the minimum attendance requirement for exams?"
    if col2.button("💰 Semester tuition fee", use_container_width=True):
        sample_query = f"What is the tuition fee structure for {student_programme}?"
    if col3.button("🎓 Scholarship criteria", use_container_width=True):
        sample_query = "What scholarships are available and how do I apply?"
else:
    sample_query = None

# Chat input box
user_input = st.chat_input("Ask about exams, credits, fees, or college policies...")
active_query = sample_query or user_input

if active_query:
    # 1. Display and store user message
    st.session_state.chat_history.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # 2. Run LangGraph pipeline and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching college records and generating response..."):
            result = app.invoke(
                {
                    "programme": student_programme,
                    "messages": [("human", active_query)],
                }
            )

            answer = result["messages"][-1].content
            category = result.get("query_type", "general")
            retrieved_context = result.get("retrieved_context", "NO_RETRIEVAL_NEEDED")

        st.caption(CATEGORY_BADGES.get(category, "💬 General Chat"))
        st.markdown(answer)

        if retrieved_context and retrieved_context != "NO_RETRIEVAL_NEEDED":
            with st.expander("🔍 View Retrieved Document Context"):
                st.text(retrieved_context)

    # 3. Save assistant response to session state
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "category": category,
            "context": retrieved_context,
        }
    )
    if sample_query:
        st.rerun()