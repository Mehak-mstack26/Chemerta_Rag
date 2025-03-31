# Chemerta_Rag

## 🚀 Retrieval-Augmented Generation (RAG) Implementation

This repository contains a **Retrieval-Augmented Generation (RAG) pipeline** designed for **chemistry-specific question answering**. The pipeline leverages **domain-specific embeddings** from ChemBERTa and integrates OpenAI's GPT model for generating accurate and contextual responses.

---

## 📌 Installation Guide

### **1️⃣ Set Up a Virtual Environment (Recommended)**

#### **Create and Activate a Virtual Environment**
```sh
python -m venv venv  # Create a virtual environment
source venv/bin/activate  # Activate on macOS/Linux
venv\Scripts\activate  # Activate on Windows
```

### **2️⃣ Install Dependencies**
#### **Using `requirements.txt`**
```sh
pip install -r requirements.txt
```

---

## 📂 Setting Up the Database

### **1️⃣ Store API Credentials in `.env` File**
Create a `.env` file in the root directory and add your **OpenAI API credentials**:
```sh
OPENAI_API_KEY="your_api_key_here"
```

### **2️⃣ Run `createDatabase.py`**
Ensure your credentials are stored in `.env` and install `python-dotenv` if not installed:
```sh
pip install python-dotenv
python createDatabase.py
```
After running the script, a `chroma` folder will be created, consisting of **vector embeddings** generated from JSON data (all reactions in the `data/` folder).

---

## 🔍 Querying Data

Once the database is created, you can run queries:

```sh
python3 query_data.py "give me the reaction involving dioxane as solvent"
```
**or**
```sh
python query_data.py "<your_query_here>"
```

This will:
1. Retrieve the most relevant document from the **Chroma vector store**.
2. Feed the retrieved data into the **chat model**.
3. Log the response for further analysis.

---

## 🎛️ Adjusting Search Parameters

### **1️⃣ Adjusting Relevance Score Threshold**
Modify the following line in `query_data.py` to increase or decrease the **relevance threshold**:
```python
if len(results) == 0 or results[0][1] < 0.1:
```
Increase `0.1` for stricter filtering, decrease it for broader results.

### **2️⃣ Adjusting the Number of Top Results**
Modify the **`k` value** in `query_data.py`:
```python
results = db.similarity_search_with_relevance_scores(query_text, k=1)
```
Change `k` to the desired number of top results from the **Chroma vector store**.

---

## 📌 Environment Variables (`.env` file)
Your `.env` file should contain the following:
```sh
OPENAI_API_KEY="your_api_key_here"
```


