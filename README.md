🔍 AI Fact-Checker

An AI-powered web application that automatically extracts factual claims from PDF documents and verifies them against live web sources.

The application is designed to help identify outdated, inaccurate, or false information in reports, research papers, and AI-generated content.

⸻

🚀 Features

* 📄 Upload PDF documents through a simple Streamlit interface
* 🧠 Automatically extract factual claims using an LLM
* 🌐 Search live web sources for supporting evidence
* ⚖️ Classify each claim as:
    * ✅ Verified
    * ⚠️ Inaccurate
    * ❌ False
    * 🕒 Outdated
    * ❓ Unverifiable
* 📊 Interactive dashboard with summary statistics
* 📥 Export verification results as CSV or JSON
* 🛡️ Input validation and error handling

⸻

🛠️ Tech Stack

* Python 3.11
* Streamlit
* LangChain
* Google Gemini
* Groq
* Tavily Search API
* DuckDuckGo Search
* PyMuPDF
* Pydantic

⸻

📂 Project Structure

fact-checker/
│
├── app/
│   ├── core/
│   │   ├── pdf_extractor.py
│   │   ├── claim_extractor.py
│   │   ├── web_searcher.py
│   │   ├── fact_verifier.py
│   │   └── pipeline.py
│   │
│   ├── models/
│   ├── ui/
│   ├── utils/
│   ├── config.py
│   └── main.py
│
├── .streamlit/
├── .env.example
├── requirements.txt
├── README.md
└── .gitignore

⸻

⚙️ Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/AI-FACT-CHECKER-FINAL.git
cd AI-FACT-CHECKER-FINAL

Create a virtual environment:

python -m venv .venv

Activate it:

Windows

.venv\Scripts\activate

macOS / Linux

source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

⸻

🔑 Configure API Keys

Copy the environment template:

cp .env.example .env

Update the .env file with your keys:

GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
TAVILY_API_KEY=your_tavily_key

⸻

▶️ Run the Application

streamlit run app/main.py

The application will be available at:

http://localhost:8501

⸻

☁️ Deployment

The application can be deployed on Streamlit Community Cloud.

Required secrets:

GROQ_API_KEY
GOOGLE_API_KEY
TAVILY_API_KEY

Main file:

app/main.py

⸻

🔄 Workflow

PDF Upload
      │
      ▼
Text Extraction
      │
      ▼
Claim Extraction
      │
      ▼
Live Web Search
      │
      ▼
Fact Verification
      │
      ▼
Results Dashboard


⸻

🎯 Project Objective

Marketing reports and AI-generated documents often contain outdated or hallucinated information. This application automatically extracts factual claims from PDF documents, verifies them using live web sources, and presents trustworthy evidence to help users identify misinformation quickly.

