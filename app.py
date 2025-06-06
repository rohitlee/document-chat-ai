import streamlit as st
import tempfile
import os
from datetime import datetime
# import plotly.express as px
import pandas as pd

from components.document_processor import DocumentProcessor
from components.nlp_processor import NLPProcessor
from components.retrieval_system import DocumentRetriever
from components.response_generator import ResponseGenerator

st.set_page_config(
    page_title="Document AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
 <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #2196f3;
        border-left: 4px solid #2196f3;
    }
    
    .bot-message {
        background-color: #9c27b0;
        border-left: 4px solid #9c27b0;
    }
    .metrics-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
 </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot_initialized' not in st.session_state:
    st.session_state.chatbot_initialized = False
    st.session_state.messages = []
    st.session_state.document_count = 0
    st.session_state.query_count = 0
    st.session_state.confidence_history = []

@st.cache_resource
def initialize_chatbot():
    """Initialize chatbot components (cached for performance)"""
    try:
        doc_processor = DocumentProcessor()
        nlp_processor = NLPProcessor()
        retriever = DocumentRetriever(doc_processor.collection)
        response_generator = ResponseGenerator()
        return doc_processor, nlp_processor, retriever, response_generator, True
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None, None, None, None, False
    
def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🤖 Document-Driven AI Chatbot</h1>
            <p>Upload documents and ask questions in multiple languages!</p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize components
    doc_processor, nlp_processor, retriever, response_generator, init_success = initialize_chatbot()
    if not init_success:
        st.error("Failed to initialize chatbot. Please refresh the page.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Document Management")

        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT files to chat with"
        )

        if uploaded_files:
            process_documents(uploaded_files, doc_processor)
        
        st.divider()

        # Language selection
        st.header("🌐 Language Settings")
        language_options = {
            "en": "English",
            "hi": "Hindi",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
        }
        selected_language = st.selectbox(
            "Choose Language",
            options=list(language_options.keys()),
            format_func=lambda x: language_options[x],
            index=0
        )

        st.divider()

        # Statistics
        st.header("📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", st.session_state.document_count)
        with col2:
            st.metric("Queries", st.session_state.query_count)
        
        if st.session_state.confidence_history:
            avg_confidence = sum(st.session_state.confidence_history) / len(st.session_state.confidence_history)
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        st.divider()

        # Quick Actions
        st.header("⚡ Quick Actions")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.query_count = 0
            st.session_state.confidence_history = []
            st.rerun()

        if st.button("Sample Questions", use_container_width=True):
            show_sample_questions()

    # Main chat interface
    st.header("💬 Chat Interface")

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        display_chat_messages()

    # Chat input
    handle_chat_input(nlp_processor, retriever, response_generator, selected_language)

def process_documents(uploaded_files, doc_processor):
    """Process uploaded documents"""
    with st.spinner("Processing documents..."):
        total_chunks = 0

        for uploaded_file in uploaded_files:
            try:
                # Save file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name

                # Process the document
                documents = doc_processor.process_document(temp_file_path)
                doc_processor.store_documents(documents)
                total_chunks += len(documents)

                # Clean up temporary file
                os.unlink(temp_file_path)

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

        if total_chunks > 0:
            st.session_state.document_count += len(uploaded_files)
            st.success(f"Successfully processed {len(uploaded_files)} document(s) into {total_chunks} chunks.")

def display_chat_messages():
    """Display chat message history"""
    if not st.session_state.messages:
        st.info("👋 Upload a document and start asking questions!")
        return

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                        <strong>You:</strong> {message['content']}
                        <br><small>{message.get('timestamp', '')}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            confidence_bar = ""
            if "confidence" in message:
                confidence = message["confidence"] * 100
                confidence_bar = f"""
                    <div style="background: #e0e0e0; border-radius: 10px; margin: 5px 0;">
                        <div style="background: linear-gradient(90deg, #4caf50 0%, #2196f3 100%); width: {confidence}%; height: 8px; border-radius: 10px;"></div>
                    </div>
                    <small>Confidence: {confidence:.1f}%</small>
                """
                sources_text = ""
                if message.get("sources"):
                    sources_text = "<br><small>Sources: " + ", ".join(message["sources"]) + "</small>"
                
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 Assistant:</strong> {message['content']}
                        {confidence_bar}
                        {sources_text}
                        <br><small>{message.get('timestamp', '')}</small>
                    </div>
                """, unsafe_allow_html=True)
            
def handle_chat_input(nlp_processor, retriever, response_generator, language):
    """Handle chat input and generate responses"""
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Add user message
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })

        # Generate response
        with st.spinner("🤔Thinking..."):
            try:
                response_data = generate_chatbot_response(user_input, nlp_processor, retriever, response_generator, language)

                # Add bot message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_data["response"],
                    "confidence": response_data["confidence"],
                    #"sources": response_data["sources"],
                    #"timestamp": timestamp
                })

                # Update statistics
                st.session_state.query_count += 1
                st.session_state.confidence_history.append(response_data["confidence"])

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Sorry, I encountered an error processing your question.",
                    "timestamp": timestamp
                })

        st.rerun()

def generate_chatbot_response(query, nlp_processor, retriever, response_generator, language):
    """Generate chatbot response"""
    # Language detection and translation
    detected_language = nlp_processor.detect_language(query)
    english_query = query
    if detected_language != "en":
        english_query = nlp_processor.translate_text(query, 'en')

    # Document retrieval
    retrieved_docs = retriever.hybrid_search(english_query, k=5)

    if not retrieved_docs:
        return {
            "response": "I couldn't find relevant information in your documents. Please try a different question.",
            "confidence": 0.0,
            "sources": []
        }
    
    # Response generation
    response = response_generator.generate_response(
        english_query, retrieved_docs, language
    )

    # Calculate confidence
    confidence = sum(doc.get('score', 0) for doc in retrieved_docs) / len(retrieved_docs)

    # Extract sources robustly
    sources = []
    for i, doc in enumerate(retrieved_docs[:3]):
        try:
            sources.append(doc.get('metadata', {}).get('source', f'Document {i+1}'))
        except Exception:
            sources.append(f'Document {i+1}')

    return {
        "response": response,
        "confidence": confidence,
        "sources": sources
    }

def show_sample_questions():
    """Display sample questions for users to try"""
    st.markdown("📝Try These Sample Questions:")
    sample_questions = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the main findings in the report?",
        "How does this document relate to [specific topic]?",
        "What are the recommendations provided?",
        "Can you explain the methodology used in this study?"
    ]

    for i, question in enumerate(sample_questions):
        if st.button(f"📝 {question}", key=f"sample_q_{i}", use_container_width=True):
            # Add question to chat input
            st.session_state.messages.append({
                "role": "user",
                "content": question,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            st.rerun()

if __name__ == "__main__":
    main()
