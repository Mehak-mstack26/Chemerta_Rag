Chemerta_Rag

🚀 Retrieval-Augmented Generation (RAG) Implementation

This repository contains a Retrieval-Augmented Generation (RAG) pipeline designed for chemistry-specific question answering. The pipeline leverages domain-specific embeddings from ChemBERTa and integrates OpenAI's GPT model for generating accurate and contextual responses.

📌 Installation Guide

1️⃣ Set Up a Virtual Environment (Recommended)

Create and Activate a Virtual Environment

python -m venv venv  # Create a virtual environment
source venv/bin/activate  # Activate on macOS/Linux
venv\Scripts\activate  # Activate on Windows

2️⃣ Install Dependencies

Using requirements.txt

pip install -r requirements.txt

Manually Installing Required Packages

pip install python-dotenv==1.0.1 \
            langchain==0.2.2 \
            langchain-community==0.2.3 \
            langchain-openai==0.1.8 \
            unstructured==0.14.4 \
            chromadb==0.5.0 \
            openai==1.31.1 \
            tiktoken==0.7.0

For macOS Users: Install onnxruntime via Conda before installing chromadb:

conda install onnxruntime -c conda-forge

📂 Setting Up the Database

1️⃣ Store API Credentials in .env File

Create a .env file in the root directory and add your OpenAI API credentials:

OPENAI_API_KEY="your_api_key_here"

2️⃣ Run createDatabase.py

Ensure your credentials are stored in .env and install python-dotenv if not installed:

pip install python-dotenv
python createDatabase.py

After running the script, a chroma folder will be created, consisting of vector embeddings generated from JSON data (all reactions in the data/ folder).

🔍 Querying Data

Once the database is created, you can run queries:

python3 query_data.py "give me the reaction involving dioxane as solvent"

or

python query_data.py "<your_query_here>"

This will:

Retrieve the most relevant document from the Chroma vector store.

Feed the retrieved data into the chat model.

Log the response for further analysis.

🎛️ Adjusting Search Parameters

1️⃣ Adjusting Relevance Score Threshold

Modify the following line in query_data.py to increase or decrease the relevance threshold:

if len(results) == 0 or results[0][1] < 0.1:

Increase 0.1 for stricter filtering, decrease it for broader results.

2️⃣ Adjusting the Number of Top Results

Modify the k value in query_data.py:

results = db.similarity_search_with_relevance_scores(query_text, k=1)

Change k to the desired number of top results from the Chroma vector store.

📌 Environment Variables (.env file)

Your .env file should contain the following:

OPENAI_API_KEY="your_api_key_here"



