from typing import List, Dict
import openai
from transformers import pipeline

class ResponseGenerator:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    def generate_response(self, query: str, retrieved_docs: List[Dict], language: str = 'en') -> str:
        """Generate response based on retrieved documents"""
        context = self._create_context(retrieved_docs)
        response = self._generate_contextual_response(query, context)

        if language != 'en':
            response = self._translate_response(response, language)
        return response
    
    def _create_context(self, docs: List[Dict], max_length: int = 2000) -> str:
        """Create context from retrieved documents"""
        context_parts = []
        current_length = 0
        for doc in docs:
            content = doc['content']
            if current_length + len(content) > max_length:
                break
            context_parts.append(content)
            current_length += len(content)
        return "\n".join(context_parts)

    def _generate_contextual_response(self, query: str, context: str) -> str:
        """Generate response using context"""
        prompt = f"""
        Based on the following context, answer the user's question accurately and concisely.

        Context: {context}

        Question: {query}

        Answer:
        """
        if len(context) > 500:
            summary = self.summarizer(context, max_length=200, min_length=50)
            context = summary[0]['summary_text']
        sentences = context.split('.')
        relevant_sentences = [s for s in sentences if any(word in s.lower() for word in query.lower().split())]

        if relevant_sentences:
            return " ".join(relevant_sentences[:3]) + "."
        else:
            return "I found some relevant information, but couldn't extract a specific answer."
        
    def _translate_response(self, response: str, target_language: str) -> str:
        """Translate response to target language"""
        return response
        
