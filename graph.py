from typing import List, Dict, Any, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv

load_dotenv()

# Define State
class GraphState(TypedDict):
    query: str
    context_chunks: List[str]
    final_answer: str
    confidence_score: float

# Setup Vector DB Retrieval
def get_retriever(persist_directory="./chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# Nodes
def retrieve(state: GraphState):
    print("---RETRIEVE---")
    query = state["query"]
    retriever = get_retriever()
    docs = retriever.invoke(query)
    context_chunks = [doc.page_content for doc in docs]
    return {"context_chunks": context_chunks, "query": query}

def generate(state: GraphState):
    print("---GENERATE---")
    query = state["query"]
    context_chunks = state.get("context_chunks", [])
    context = "\n\n".join(context_chunks)
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    system = """You are an AI assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question strictly based on the context.
    If you don't know the answer or the context doesn't contain the answer, say that the information is not available in the eBook context, and return a confidence score of 0.0.
    Output your response in JSON format with two keys:
    - "answer": the generated answer
    - "confidence_score": a float between 0.0 and 1.0 indicating how confident you are that the answer is grounded in the context.
    
    Context:
    {context}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{query}")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        response = chain.invoke({"context": context, "query": query})
        answer = response.get("answer", "I don't know.")
        score = float(response.get("confidence_score", 0.0))
    except Exception as e:
        print(f"Error during LLM generation: {e}")
        answer = f"Error generating answer: {str(e)}"
        score = 0.0
        
    return {"final_answer": answer, "confidence_score": score, "context_chunks": context_chunks, "query": query}

# Build Graph
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app_graph = workflow.compile()
