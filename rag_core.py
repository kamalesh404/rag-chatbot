import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# Directory for ChromaDB persistence
CHROMA_PATH = "chroma_db"

def get_vector_store():
    """Initializes or loads the Chroma vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )
    return vector_store

def process_and_add_document(file_path):
    """Loads a document, splits it into chunks, and adds it to the vector store."""
    # 1. Load Document
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")
    
    docs = loader.load()
    
    # 2. Split Document
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    
    # 3. Add to Vector Store
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    
    return len(chunks)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_qa_chain():
    """Creates a Retrieval-Augmented Generation (RAG) chain using LCEL."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )
    
    vector_store = get_vector_store()
    # Retrieve top 4 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # System prompt for the LLM
    prompt = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Keep the answer concise and relevant.\n\n"
        "Context: {context}\n\n"
        "Question: {question}"
    )
    
    # Construct the LCEL chain
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # This chain will return a dictionary with "context", "question", and "answer"
    rag_chain = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    
    return rag_chain
