import os
import numpy as np
import PyPDF2
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq

class FinancialRAGPipeline:
    def __init__(self, pinecone_api_key, groq_api_key):
        self.pinecone_api_key = pinecone_api_key
        self.groq_api_key = groq_api_key
        self.index_name = 'financial-rag'
        
        # Initialize models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.groq_client = Groq(api_key=self.groq_api_key)
        
        # Initialize Pinecone
        self._init_pinecone()
    
    def _init_pinecone(self):
        """Initialize Pinecone index"""
        pc = Pinecone(api_key=self.pinecone_api_key)
        
        # Create index if not exists
        if self.index_name not in [index.name for index in pc.list_indexes()]:
            pc.create_index(
                name=self.index_name, 
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        self.index = pc.Index(self.index_name)
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from uploaded PDF"""
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
            return full_text
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""
    
    def preprocess_text(self, text):
        """Chunk text for embedding"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        return text_splitter.split_text(text)
    
    def generate_embeddings(self, chunks):
        """Generate vector embeddings"""
        return self.embedding_model.encode(chunks)
    
    def store_document_vectors(self, chunks, embeddings):
        """Store document vectors in Pinecone"""
        vectors = [
            (str(i), embedding.tolist(), {"text": chunk})
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        # Batch upsert
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(batch)
    
    def retrieve_relevant_context(self, query, top_k=5):
        """Semantic search for relevant context"""
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.index.query(
            vector=query_embedding.tolist(), 
            top_k=top_k, 
            include_metadata=True
        )
        
        return [
            {
                'text': match['metadata']['text'],
                'score': match['score']
            } 
            for match in results['matches']
        ]
    
    def generate_response(self, query, contexts):
        """Generate AI-powered response"""
        try:
            context_text = " ".join([ctx['text'] for ctx in contexts])
            
            prompt = f"""
            Financial Context: {context_text}
            
            Question: {query}
            
            Provide a precise, professional financial analysis based on the given context. 
            If specific information is unavailable, clearly state the limitations.
            
            Include:
            - Direct answer to the query
            - Relevant financial insights
            - Context-based explanation
            """
            
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional financial analyst specializing in P&L statement interpretation."
                    },
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",
                max_tokens=300,
                temperature=0.3
            )
            
            return {
                'response': chat_completion.choices[0].message.content,
                'contexts': contexts
            }
        
        except Exception as e:
            return {
                'response': f"Error generating response: {str(e)}",
                'contexts': contexts
            }
    
    def process_document(self, pdf_file):
        """Full document processing pipeline"""
        # Extract text
        pdf_text = self.extract_text_from_pdf(pdf_file)
        
        # Preprocess
        chunks = self.preprocess_text(pdf_text)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Store vectors
        self.store_document_vectors(chunks, embeddings)
        
        return len(chunks)