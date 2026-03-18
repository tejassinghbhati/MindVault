import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import ollama
from dotenv import load_dotenv

load_dotenv()

class AIService:
    def __init__(self):
        # Initialize Embedding Model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = os.getenv("CHROMA_PORT", "8000")
        self.chroma_client = chromadb.HttpClient(host=chroma_host, port=int(chroma_port))
        
        # Ensure collection exists
        self.collection = self.chroma_client.get_or_create_collection(name="mindvault_notes")
        
        # Ollama config
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

    def get_embeddings(self, text: str):
        return self.embedding_model.encode(text).tolist()

    def add_to_vector_store(self, chunk_id: str, text: str, metadata: dict):
        embedding = self.get_embeddings(text)
        self.collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata]
        )

    def delete_from_vector_store(self, chunk_ids: list):
        self.collection.delete(ids=chunk_ids)

    def query_vector_store(self, query_text: str, n_results: int = 5):
        query_embedding = self.get_embeddings(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results

    def generate_answer(self, query: str, context: str):
        prompt = f"""
        You are MindVault Assistant. Answer the user's question based ONLY on the provided context from their notes.
        If the context does not contain the answer, say you don't know based on the notes.
        
        Context:
        {context}
        
        Question: {query}
        
        Answer:
        """
        response = ollama.generate(model=self.ollama_model, prompt=prompt)
        return response['response']

ai_service = AIService()
