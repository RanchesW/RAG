import streamlit as st
import sys
import logging
from pathlib import Path
from typing import Optional, List

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

# Setup logging
logging.basicConfig(level=logging.WARNING)

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system - cached to avoid reloading"""
    try:
        from src.rag.pipeline import KazakhstanRAGPipeline
        
        config = {
            "embedding_model": settings.EMBEDDING_MODEL,
            "llm_model": settings.LLM_MODEL,
            "qdrant_path": str(settings.QDRANT_DIR),
            "collection_name": settings.QDRANT_COLLECTION,
            "tensor_parallel_size": settings.TENSOR_PARALLEL_SIZE
        }
        
        with st.spinner("🔄 Loading Kazakhstan RAG System..."):
            pipeline = KazakhstanRAGPipeline(config)
        
        return pipeline, None
        
    except Exception as e:
        return None, str(e)

def setup_directories():
    """Create required directories"""
    dirs = [settings.QDRANT_DIR, settings.UPLOADS_DIR, settings.DATA_DIR, settings.LOGS_DIR]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

def upload_and_index_document(pipeline, uploaded_file):
    """Handle single document upload and indexing"""
    try:
        # Save uploaded file
        upload_path = settings.UPLOADS_DIR / uploaded_file.name
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Index document
        success = pipeline.index_document(str(upload_path))
        return success, uploaded_file.name
        
    except Exception as e:
        return False, f"{uploaded_file.name}: {str(e)}"

def upload_and_index_multiple_documents(pipeline, uploaded_files):
    """Handle multiple documents upload and indexing"""
    results = []
    
    # Create progress tracking
    progress_container = st.container()
    
    with progress_container:
        st.subheader("📤 Processing Files")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each file
        for i, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
            
            # Process file
            try:
                # Save uploaded file
                upload_path = settings.UPLOADS_DIR / uploaded_file.name
                with open(upload_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Index document
                success = pipeline.index_document(str(upload_path))
                
                results.append({
                    "file_name": uploaded_file.name,
                    "status": "✅ Success" if success else "❌ Failed",
                    "success": success,
                    "size": len(uploaded_file.getvalue()),
                    "error": None
                })
                
            except Exception as e:
                results.append({
                    "file_name": uploaded_file.name,
                    "status": "❌ Error",
                    "success": False,
                    "size": len(uploaded_file.getvalue()) if hasattr(uploaded_file, 'getvalue') else 0,
                    "error": str(e)
                })
        
        # Final progress
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
    
    return results

def main():
    st.title("🇰🇿 Kazakhstan RAG System")
    st.markdown("*Corporate Document Assistant for Kazakhstan*")
    
    # Setup directories
    setup_directories()
    
    # Initialize RAG system
    pipeline, error = initialize_rag_system()
    
    if error:
        st.error(f"❌ Failed to initialize system: {error}")
        st.info("Make sure all dependencies are installed: `pip install -r requirements.txt`")
        return
    
    if not pipeline:
        st.error("❌ System not ready")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # Upload method selection
        upload_method = st.radio(
            "Upload Method:",
            ["Single File", "Multiple Files", "Directory"],
            help="Choose how you want to upload documents"
        )
        
        if upload_method == "Single File":
            # Single file upload
            uploaded_file = st.file_uploader(
                "Upload Document",
                type=['pdf', 'docx', 'pptx', 'xlsx', 'txt'],
                help="Upload a single corporate document to index"
            )
            
            if uploaded_file and st.button("📤 Upload & Index"):
                with st.spinner(f"📚 Indexing {uploaded_file.name}..."):
                    success, message = upload_and_index_document(pipeline, uploaded_file)
                
                if success:
                    st.success(f"✅ Successfully indexed '{uploaded_file.name}'!")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to index: {message}")
        
        elif upload_method == "Multiple Files":
            # Multiple files upload
            uploaded_files = st.file_uploader(
                "Upload Multiple Documents",
                type=['pdf', 'docx', 'pptx', 'xlsx', 'txt'],
                accept_multiple_files=True,
                help="Upload multiple corporate documents at once"
            )
            
            if uploaded_files:
                st.info(f"📋 Selected {len(uploaded_files)} files")
                
                # Show file list
                with st.expander("📄 File List", expanded=False):
                    for i, file in enumerate(uploaded_files, 1):
                        file_size = len(file.getvalue()) / 1024  # KB
                        st.write(f"{i}. **{file.name}** ({file_size:.1f} KB)")
                
                if st.button("📤 Upload & Index All", type="primary"):
                    results = upload_and_index_multiple_documents(pipeline, uploaded_files)
                    
                    # Show results
                    st.subheader("📊 Processing Results")
                    
                    # Summary stats
                    successful = sum(1 for r in results if r["success"])
                    failed = len(results) - successful
                    total_size = sum(r["size"] for r in results) / (1024 * 1024)  # MB
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("✅ Successful", successful)
                    with col2:
                        st.metric("❌ Failed", failed)
                    with col3:
                        st.metric("📁 Total Size", f"{total_size:.1f} MB")
                    
                    # Detailed results
                    st.subheader("📄 File Details")
                    for result in results:
                        with st.expander(f"{result['status']} {result['file_name']}", 
                                       expanded=not result["success"]):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Status:** {result['status']}")
                                st.write(f"**Size:** {result['size'] / 1024:.1f} KB")
                            with col2:
                                if result["error"]:
                                    st.error(f"**Error:** {result['error']}")
                                else:
                                    st.success("Successfully indexed")
                    
                    if successful > 0:
                        st.success(f"🎉 Successfully processed {successful}/{len(results)} files!")
                        st.rerun()
        
        else:  # Directory upload
            st.subheader("📚 Bulk Index from Directory")
            doc_directory = st.text_input("Document Directory Path", placeholder="/path/to/documents")
            
            if st.button("🗂️ Index Directory") and doc_directory:
                if Path(doc_directory).exists():
                    with st.spinner("Scanning directory..."):
                        try:
                            doc_dir = Path(doc_directory)
                            supported_extensions = ['.pdf', '.docx', '.pptx', '.xlsx', '.txt']
                            files = [f for f in doc_dir.rglob("*") 
                                   if f.is_file() and f.suffix.lower() in supported_extensions]
                            
                            if not files:
                                st.warning("No supported documents found")
                            else:
                                st.info(f"Found {len(files)} documents")
                                
                                # Process with progress
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                success_count = 0
                                results = []
                                
                                for i, file_path in enumerate(files):
                                    progress = (i + 1) / len(files)
                                    progress_bar.progress(progress)
                                    status_text.text(f"Processing: {file_path.name} ({i+1}/{len(files)})")
                                    
                                    try:
                                        success = pipeline.index_document(str(file_path))
                                        if success:
                                            success_count += 1
                                            results.append(f"✅ {file_path.name}")
                                        else:
                                            results.append(f"❌ {file_path.name}")
                                    except Exception as e:
                                        results.append(f"❌ {file_path.name}: {e}")
                                
                                # Show results
                                st.success(f"✅ Indexed {success_count}/{len(files)} documents")
                                
                                with st.expander("📄 Detailed Results"):
                                    for result in results:
                                        st.write(result)
                                
                                if success_count > 0:
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Directory indexing failed: {e}")
                else:
                    st.error("Directory not found")
        
        st.divider()
        
        # Document list and deletion
        st.subheader("📋 Manage Documents")
        
        # Get list of documents
        documents = pipeline.get_all_documents()
        
        if documents:
            st.write(f"📊 **{len(documents)} documents indexed**")
            
            # Quick stats
            total_chunks = sum(doc['chunks'] for doc in documents)
            st.caption(f"Total chunks: {total_chunks}")
            
            # Delete all button
            if st.button("🗑️ Delete ALL Documents", type="secondary"):
                if st.session_state.get('confirm_delete_all', False):
                    with st.spinner("Deleting all documents..."):
                        success = pipeline.delete_all_documents()
                    if success:
                        st.success("✅ All documents deleted!")
                        st.session_state.confirm_delete_all = False
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete documents")
                else:
                    st.session_state.confirm_delete_all = True
                    st.warning("⚠️ Click again to confirm deletion of ALL documents")
            
            st.divider()
            
            # Individual document deletion
            st.write("**Individual Documents:**")
            
            # Sort documents by name
            sorted_docs = sorted(documents, key=lambda x: x['file_name'].lower())
            
            for doc in sorted_docs:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"📄 **{doc['file_name']}**")
                    st.caption(f"{doc['chunks']} chunks")
                
                with col2:
                    delete_key = f"delete_{doc['file_name']}"
                    if st.button("🗑️", key=delete_key, help=f"Delete {doc['file_name']}"):
                        with st.spinner(f"Deleting {doc['file_name']}..."):
                            success = pipeline.delete_document(doc['file_name'])
                        if success:
                            st.success(f"✅ Deleted {doc['file_name']}")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to delete {doc['file_name']}")
        else:
            st.info("No documents indexed yet")
        
        st.divider()
        
        # System stats
        if st.button("📊 System Stats"):
            try:
                stats = pipeline.get_system_stats()
                st.json(stats)
            except Exception as e:
                st.error(f"Failed to get stats: {e}")
    
    # Main chat interface
    st.header("💬 Ask Questions")
    
    # Show quick stats in main area
    if pipeline:
        stats = pipeline.get_system_stats()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Documents", stats.get("documents_indexed", 0))
        with col2:
            st.metric("🤖 LLM Status", "Ready" if stats.get("llm_loaded") else "Not Ready")
        with col3:
            st.metric("📊 Collection", stats.get("collection_status", "Unknown"))
    
    # Query input
    query = st.text_input(
        "Your Question (Russian/English):",
        placeholder="Задайте вопрос о документах...",
        help="Ask questions about your uploaded documents in Russian or English"
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Number of sources", 1, 10, 5)
            show_sources = st.checkbox("Show source details", True)
        with col2:
            prompt_style = st.selectbox("Response style", ["detailed", "concise"])
    
    # Query button
    if st.button("🔍 Search", disabled=not query or not pipeline):
        if query:
            with st.spinner("Searching documents and generating answer..."):
                try:
                    result = pipeline.query(
                        question=query,
                        top_k=top_k
                    )
                    
                    if result:
                        # Display answer
                        st.subheader("📝 Answer")
                        st.markdown(result["answer"])
                        
                        # Display confidence and metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            confidence = result.get("confidence", 0)
                            st.metric("Confidence", f"{confidence:.2%}")
                        with col2:
                            context_used = result.get("context_used", 0)
                            st.metric("Sources Used", context_used)
                        
                        # Display sources
                        if show_sources and result.get("sources"):
                            st.subheader("📚 Sources")
                            for i, source in enumerate(result["sources"], 1):
                                with st.expander(f"Source {i}: {source['file_name']} (Score: {source['score']:.3f})"):
                                    st.write(f"**File:** {source['file_name']}")
                                    st.write(f"**Relevance Score:** {source['score']:.3f}")
                                    if source['score'] > 0.7:
                                        st.success("High relevance")
                                    elif source['score'] > 0.5:
                                        st.info("Medium relevance")
                                    else:
                                        st.warning("Low relevance")
                    else:
                        st.error("No results found")
                        
                except Exception as e:
                    st.error(f"❌ Query failed: {str(e)}")

    # Instructions
    st.divider()
    st.subheader("📖 How to Use")
    st.markdown("""
    ### Upload Options:
    - **Single File**: Upload one document at a time
    - **Multiple Files**: Select and upload multiple documents simultaneously  
    - **Directory**: Index all supported files from a folder path
    
    ### Supported File Types:
    PDF, DOCX, PPTX, XLSX, TXT
    
    ### Features:
    - **Batch Processing**: Upload multiple files with progress tracking
    - **Document Management**: View and delete individual documents or all at once
    - **Real-time Stats**: Monitor indexed documents and system status
    - **Semantic Search**: Ask questions in Russian or English and get contextual answers
    """)

if __name__ == "__main__":
    main()
