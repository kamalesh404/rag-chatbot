import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from rag_core import process_and_add_document, get_qa_chain

# Load environment variables
load_dotenv()

st.set_page_config(page_title="RAG AI Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Chat with Your Documents")
st.caption("Powered by Gemini, LangChain, and ChromaDB")

# Check for API Key
if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_api_key_here":
    st.warning("Please set your GOOGLE_API_KEY in the `.env` file to use this application.")
    st.stop()

# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=['pdf', 'txt'])
    
    if st.button("Process Document"):
        if uploaded_file is not None:
            with st.spinner("Processing document..."):
                # Save uploaded file temporarily to disk so LangChain can load it
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    chunks_added = process_and_add_document(tmp_path)
                    st.success(f"Successfully processed and added {chunks_added} chunks to the knowledge base!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    os.unlink(tmp_path) # Clean up temp file
        else:
            st.error("Please upload a file first.")
            
    st.divider()
    st.markdown("### 💾 Project Location")
    st.info(f"You can delete this project later at:\n`{os.path.abspath(os.getcwd())}`")


# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for idx, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {idx + 1}:**")
                    st.text(source.page_content)

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                qa_chain = get_qa_chain()
                response = qa_chain.invoke({"input": prompt})
                
                answer = response["answer"]
                sources = response.get("context", [])
                
                st.markdown(answer)
                
                if sources:
                    with st.expander("View Sources"):
                        for idx, source in enumerate(sources):
                            st.markdown(f"**Source {idx + 1}:**")
                            st.text(source.page_content)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"Error generating answer: {e}")
