import os 
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph , START, END
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-120b" , temperature=0.2 )
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def retriver(pdf_path :str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever(search_kwargs={"k": 4})



academics_handbook= retriver("academics_handbook.pdf")
fee_structure = retriver("fee_structure.pdf")


class stat(TypedDict):
    program : str
    message : Annotated[list, add_messages]
    quary_type:str
    retriver_context: str

    





