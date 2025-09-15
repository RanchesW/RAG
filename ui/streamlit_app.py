import streamlit as st
import requests
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings

# Page config
st.set_page_config(
    page_title="Kazakhstan RAG System",
    page_icon="🇰🇿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE = f"http://{settings.API_HOST}:{settings.API_PORT}"

def main():
    st.title("🇰🇿 Kazakhstan RAG System")
    st.markdown("*Corporate Document Assistant for Kazakhstan*")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'docx', 'pptx', 'xlsx', 'txt'],
            help="Upload corporate documents to index"
        )
        
        if uploaded_file and st.button("📤 Upload & Index"):
            with st.spinner("Uploading and indexing document..."):
                try:
                    files = {"file": uploaded_file.getvalue()}
                    response = requests.post(f"{API_BASE}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success(f"✅ Document '{uploaded_file.name}' indexed successfully!")
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        
        # System stats
        if st.button("📊 System Stats"):
            try:
                response = requests.get(f"{API_BASE}/stats")
                if response.status_code == 200:
                    stats = response.json()
                    st.json(stats)
                else:
                    st.error("Failed to get stats")
            except:
                st.error("API not available")
    
    # Main chat interface
    st.header("💬 Ask Questions")
    
    # Query input
    query = st.text_input(
        "Your Question (Russian/English):",
        placeholder="Задайте вопрос о документах...",
        help="Ask questions about your uploaded documents in Russian or English"
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        top_k = st.slider("Number of sources", 1, 10, 5)
        show_sources = st.checkbox("Show source details", True)
    
    # Query button
    if st.button("🔍 Search", disabled=not query):
        if query:
            with st.spinner("Searching documents and generating answer..."):
                try:
                    payload = {
                        "question": query,
                        "top_k": top_k
                    }
                    
                    response = requests.post(
                        f"{API_BASE}/query",
                        json=payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display answer
                        st.subheader("📝 Answer")
                        st.write(result["answer"])
                        
                        # Display confidence
                        confidence = result["confidence"]
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Display sources
                        if show_sources and result["sources"]:
                            st.subheader("📚 Sources")
                            for i, source in enumerate(result["sources"], 1):
                                with st.expander(f"Source {i}: {source['file_name']}"):
                                    st.write(f"**Relevance Score:** {source['score']:.3f}")
                        
                        # Context info
                        st.info(f"Used {result['context_used']} document chunks for this answer")
                        
                    else:
                        st.error(f"❌ Query failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Make sure the API server is running: `python main.py api`")

    # Instructions
    st.divider()
    st.subheader("📖 How to Use")
    st.markdown("""
    1. **Upload Documents**: Use the sidebar to upload PDF, Word, PowerPoint, Excel, or text files
    2. **Ask Questions**: Type your question in Russian or English
    3. **Get Answers**: The system will search your documents and provide answers with sources
    
    **Supported file types**: PDF, DOCX, PPTX, XLSX, TXT
    """)

if __name__ == "__main__":
    main()
