from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

class DocumentRetriever:
    def __init__(self, chroma_collection):
        self.collection = chroma_collection
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform similarity search on documents"""
        query_embedding = self.embedding_model.encode(query)
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )

        return self._format_results(results)
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        """Combine semantic and keyword search"""
        semantic_results = self.similarity_search(query, k)
        keyword_results = self._keyword_search(query, k)
        combined_results = self._merge_results(semantic_results, keyword_results)

        return combined_results[:k]
    
    def _keyword_search(self, query: str, k: int) -> List[Dict]:
        """Simple keyword-based search"""
        keywords = query.lower().split()
        results = []

        all_docs = self.collection.get()
        for i, doc in enumerate(all_docs['documents']):
            score = sum(1 for keyword in keywords if keyword in doc.lower())
            if score > 0:
                results.append({
                    'content': doc,
                    'score': score,
                    'metadata': all_docs['metadatas'][i]  # <-- fixed here
                })
        return sorted(results, key=lambda x: x['score'], reverse=True)[:k]

    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results"""
        formatted = []
        for i in range(len(results['documents'])):
            formatted.append({
                'content': results['documents'][0][i],
                'score': 1 - results['distances'][0][i], 
                'metadata': results['metadatas'][0][i]
            })
        return formatted
    
    def _merge_results(self, semantic: List[Dict], keyword: List[Dict]) -> List[Dict]:
        """Merge semantic and keyword search results, preserving metadata"""
        combined = {}
        metadata_lookup = {}

        # Build a lookup for metadata from all docs in the collection
        all_docs = self.collection.get()
        for i, doc in enumerate(all_docs['documents']):
            metadata_lookup[doc] = all_docs['metadatas'][i]

        for result in semantic:
            content = result['content']
            combined[content] = combined.get(content, 0) + result['score'] * 0.7

        for result in keyword:
            content = result['content']
            combined[content] = combined.get(content, 0) + result['score'] * 0.3

        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        # Return with metadata
        return [
            {
                'content': content,
                'score': score,
                'metadata': metadata_lookup.get(content, {})
            }
            for content, score in sorted_results
        ]
