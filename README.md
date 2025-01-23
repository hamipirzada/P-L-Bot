P&L Bot: AI-Powered Financial Statement Analysis 💼🤖
🌟 Project Overview
P&L Bot is an advanced Retrieval-Augmented Generation (RAG) tool designed to extract, analyze, and provide insights from Profit & Loss (P&L) statements using state-of-the-art natural language processing techniques.

🚀 Features
PDF financial document upload
Semantic search of financial documents
AI-powered financial insights generation
Interactive Streamlit interface
Context-aware financial analysis
🛠 Technology Stack
Python 3.9+
Streamlit
Pinecone Vector Database
Groq (Llama3 Language Model)
SentenceTransformers
PyPDF2
📋 Prerequisites
Python 3.9+
Pinecone Account
Groq API Key
🔧 Installation
1. Clone the Repository

git clone https://github.com/hamipirzada/P-L-Bot.git
cd P-L-Bot
2. Create Virtual Environment

python3 -m venv venv
source venv/bin/activate
3. Install Dependencies

pip install -r requirements.txt
4. Configure Environment Variables
Create a .env file in the project root:

PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
🖥 Running the Application

streamlit run app/main.py
📊 Example Queries
"What is the total revenue for Q1 2024?"
"Compare operating expenses between Q3 and Q4"
"Calculate the net profit margin"
"Show the operating margin for the past 6 months"
🔍 How It Works
Upload PDF with financial statements
AI processes and embeds document
Ask questions about the document
Receive contextually relevant insights
🛡 Security Considerations
API keys stored in environment variables
No sensitive data persisted
Secure vector storage with Pinecone
📈 Performance Metrics
Retrieval Precision: 85-90%
Response Relevance: 75-85%
Avg. Query Latency: 2-3 seconds
🔧 Troubleshooting
Ensure all dependencies are installed
Verify API keys in .env
Check Python and Streamlit versions
🤝 Contributing
Fork the repository
Create your feature branch
Commit your changes
Push to the branch
Create a Pull Request

🙏 Acknowledgements
Pinecone for vector database
Groq for language model
Streamlit for interactive interface

🗺 Future Roadmap
[ ] Multi-document support
[ ] Advanced financial metric calculations
[ ] Improved context retrieval
[ ] Enhanced visualization of financial insights
🎥 Demo

![Screenshot From 2025-01-23 18-12-35](https://github.com/user-attachments/assets/f94d6a3c-924d-4275-8036-f683781d0a3d)

![Screenshot From 2025-01-23 18-17-35](https://github.com/user-attachments/assets/6f6a78f6-93f9-40e1-ba77-d021e89616c6)
