import streamlit as st
from streamlit_chat import message
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceEndpoint
from langchain.memory import ConversationBufferMemory
from pathlib import Path


load_dotenv()
api_token = os.getenv("HF_TOKEN")



DATA_FOLDER = "data"  # folder containing PDF files
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

def get_pdf_files():
    """Get all PDF files from the data folder"""
    data_path = Path(DATA_FOLDER)
    if not data_path.exists():
        st.error(f"Data folder '{DATA_FOLDER}' not found. Please create it and add PDF files.")
        return []
    return list(data_path.glob("*.pdf"))

def initialize_session_state():
    """Initialize session state variables"""
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'qa_chain' not in st.session_state:
        st.session_state['qa_chain'] = None
    if 'files_processed' not in st.session_state:
        st.session_state['files_processed'] = []

def create_vectordb():
    """Create vector database from all PDF documents in the data folder"""
    pdf_files = get_pdf_files()
    
    if not pdf_files:
        st.error("No PDF files found in the data folder.")
        return None
    
    
    loaders = [PyPDFLoader(str(pdf_file)) for pdf_file in pdf_files]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    
    
    st.session_state['files_processed'] = [pdf_file.name for pdf_file in pdf_files]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64
    )
    splits = text_splitter.split_documents(pages)
    
    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(splits, embeddings)
    return vectordb

def initialize_qa_chain():
    """Initialize the QA chain with the LLM and vector database"""
    if st.session_state['qa_chain'] is None:
        vectordb = create_vectordb()
        if vectordb is None:
            return False
        
        llm = HuggingFaceEndpoint(
            repo_id=MODEL_NAME,
            huggingfacehub_api_token=api_token,
            temperature=0.5,
            max_new_tokens=4096,
            top_k=3,
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key='answer',
            return_messages=True
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectordb.as_retriever(),
            chain_type="stuff",
            memory=memory,
            return_source_documents=True,
            verbose=False,
        )
        
        st.session_state['qa_chain'] = qa_chain
        return True
    return True

def format_chat_history():
    """Format chat history for the LLM"""
    formatted_history = []
    for user_msg, bot_msg in zip(st.session_state['past'], st.session_state['generated']):
        formatted_history.append(f"User: {user_msg}")
        formatted_history.append(f"Assistant: {bot_msg}")
    return formatted_history

def generate_response(user_input):
    """Generate response using the QA chain"""
    formatted_history = format_chat_history()
    
    response = st.session_state['qa_chain'].invoke({
        "question": user_input,
        "chat_history": formatted_history
    })
    
    answer = response["answer"]
    if "Helpful Answer:" in answer:
        answer = answer.split("Helpful Answer:")[-1]
        
    return answer, response["source_documents"][:3]

def main():
    st.title("Multi-Document Chat Assistant")
    
    initialize_session_state()
    
    
    with st.sidebar:
        st.markdown("### Processed Files")
        if st.session_state['files_processed']:
            for file in st.session_state['files_processed']:
                st.write(f"📄 {file}")
        else:
            st.write("No files processed yet")
        
        if st.button("Reset Conversation"):
            st.session_state['generated'] = []
            st.session_state['past'] = []
            st.session_state['qa_chain'] = None
            st.session_state['files_processed'] = []
            st.experimental_rerun()
    
    
    if st.session_state['qa_chain'] is None:
        with st.spinner("Initializing chat assistant and processing documents..."):
            if initialize_qa_chain():
                st.success("Chat assistant is ready!")
            else:
                st.error("Failed to initialize chat assistant. Please check the data folder.")
                return
    
    
    user_input = st.text_input("Ask a question about your documents:", key="input")
    
    if user_input:
        with st.spinner("Thinking..."):
            answer, sources = generate_response(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(answer)
    
    
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state['generated'][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            
            
            if i == len(st.session_state['generated'])-1:
                with st.expander("View Sources"):
                    for idx, source in enumerate(sources, 1):
                        source_file = Path(source.metadata['source']).name
                        st.markdown(f"**Source {idx} (File: {source_file}, Page {source.metadata['page'] + 1}):**")
                        st.text(source.page_content.strip())
                        st.markdown("---")

if __name__ == "__main__":
    main()