import os
import streamlit as st
from dotenv import load_dotenv
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

# Import RAG Pipeline
from rag_pipeline import FinancialRAGPipeline

# Streamlit Page Configuration
st.set_page_config(
    page_title="Financial P&L QA Bot",
    page_icon="💼",
    layout="wide"
)

# Initialize RAG Pipeline
@st.cache_resource
def init_rag_pipeline():
    # Retrieve API keys from environment variables
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    # Validate API keys
    if not pinecone_api_key or not groq_api_key:
        st.error("Please set PINECONE_API_KEY and GROQ_API_KEY in your .env file")
        return None
    
    return FinancialRAGPipeline(
        pinecone_api_key=pinecone_api_key,
        groq_api_key=groq_api_key
    )

def main():
    st.title("🤖 Financial P&L Question Answering Bot")
    
    # Initialize RAG Pipeline
    rag_pipeline = init_rag_pipeline()
    
    if rag_pipeline is None:
        return
    
    # Sidebar for Document Upload
    st.sidebar.header("Upload Financial Document")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF", 
        type=['pdf'], 
        help="Upload your Profit & Loss Statement PDF"
    )
    
    # Document Processing
    if uploaded_file is not None:
        with st.spinner('Processing Document...'):
            chunk_count = rag_pipeline.process_document(uploaded_file)
            st.sidebar.success(f"Processed {chunk_count} document chunks")
    
    # Query Interface
    st.header("Ask a Financial Question")
    query = st.text_input(
        "Enter your financial query", 
        placeholder="e.g., What is the total revenue for Q1 2024?"
    )
    
    # Query Button
    if st.button("Get Insights") and query:
        with st.spinner('Retrieving Insights...'):
            # Retrieve and Generate Response
            result = rag_pipeline.generate_response(
                query, 
                rag_pipeline.retrieve_relevant_context(query)
            )
        
        # Display Response
        st.subheader("AI Analysis")
        st.write(result['response'])
        
        # Display Retrieved Contexts
        st.subheader("Retrieved Context Segments")
        for i, context in enumerate(result['contexts'], 1):
            st.markdown(f"**Context {i}** (Relevance Score: {context['score']:.2f})")
            st.code(context['text'], language='text')

if __name__ == "__main__":
    main()