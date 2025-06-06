import os
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from sentence_transformers import SentenceTransformer
import chromadb

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection("documents")

    def process_document(self, file_path: str) -> List[Dict]:
        """Load and process documents"""
        documents = []

        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)

        for i, chunk in enumerate(chunks):
            doc_data = {
                'id': f"{file_path}_{i}",
                'content': chunk.page_content,
                'metadata': chunk.metadata,
                'embedding': self.embedding_model.encode(chunk.page_content)
            }
            documents.append(doc_data)

        return documents
    
    def store_documents(self, documents: List[Dict]):
        """Store documents in vector database"""
        for doc in documents:
            self.collection.add(
                ids=[doc['id']],
                embeddings=[doc['embedding'].tolist()],
                documents=[doc['content']],
                metadatas=[doc['metadata']]
            )