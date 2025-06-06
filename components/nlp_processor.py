import spacy
from transformers import pipeline
import requests

class NLPProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.intent_classifier = pipeline("text-classification", model="microsoft/DialoGPT-medium")
        self.sarvam_api_key = "Your_Sarvam_API_Key"  # Replace with your Sarvam API key

    def detect_language(self, text: str) -> str:
        """Detect language using Sarvam AI"""
        headers = {
            'Authorization': f'Bearer {self.sarvam_api_key}',
            'Content-Type': 'application/json'
        }

        response = requests.post('https://api.sarvam.ai/v1/detect_language', headers=headers, json={'text': text})
        return response.json().get('language', 'en')
    
    def translate_text(self, text: str, target_lang: str) -> str:
        """Translate text using Sarvam AI"""
        if target_lang == 'en':
            return text
        
        headers = {
            'Authorization': f'Bearer {self.sarvam_api_key}',
            'Content-Type': 'application/json'
        }

        response = requests.post('https://api.sarvam.ai/v1/translate', headers=headers, json={'text': text, 'target_language': target_lang})
        return response.json().get('translated_text', text)
    
    def extract_entities(self, text: str) -> dict:
        """Extract named entities from text"""
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return {'entities': entities, 'tokens': [token.text for token in doc]}

