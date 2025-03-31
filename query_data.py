import argparse
import json
import os
import re
from dotenv import load_dotenv
from thefuzz import fuzz
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.preprocessing import normalize
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
CHROMA_PATH = "chroma"
BASE_DATA_PATH = "/Users/mehakjain/Desktop/rag model/data/"

if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

prompt_template = PromptTemplate(
    input_variables=["context", "input"],
    template="""
You are a chemistry assistant. You will be provided with a JSON-formatted chemical reaction. 
Your task is to extract the answer to the user's question directly from the JSON.

JSON:
{context}

Question: {input}

Answer:
Respond only with the exact value, list of values, or full JSON block if required.
Do not add any extra text or explanation.
"""
)


class ChemBERTaEmbeddings(Embeddings):
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def _get_cls_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
        return cls_embedding.squeeze().numpy()

    def embed_documents(self, texts):
        embeddings = [self._get_cls_embedding(text) for text in texts]
        normalized = normalize(np.array(embeddings))
        return normalized.tolist()

    def embed_query(self, text):
        embedding = self._get_cls_embedding(text)
        normalized = normalize([embedding])[0]
        return normalized.tolist()  # returns a 1D list as required


def ask_openai(context: str, question: str) -> str:
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4-0125-preview",
        openai_api_key=api_key
    )
    prompt = prompt_template.format(context=context, input=question)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip().strip('"')


def extract_reaction_names_from_query(query, threshold=80):
    """Fuzzy match reaction names from query using known reactions from the dataset"""
    available_files = list_available_reaction_files()
    known_reactions = [name for name, _ in available_files]

    matched_reactions = []
    for reaction in known_reactions:
        score = fuzz.partial_ratio(reaction.lower(), query.lower())
        if score >= threshold:
            matched_reactions.append((reaction, score))

    # Sort by score descending and return just the names
    matched_reactions.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in matched_reactions]


def list_available_reaction_files():
    """List all JSON reaction files in the data directory"""
    files = []
    for filename in os.listdir(BASE_DATA_PATH):
        if filename.endswith('.json'):
            try:
                filepath = os.path.join(BASE_DATA_PATH, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    if "reactionName" in data:
                        files.append((data["reactionName"], filepath))
            except (json.JSONDecodeError, IOError):
                continue
    return files


def find_reaction_file_by_name(reaction_name):
    """Find a reaction file by its exact name or partial match"""
    available_files = list_available_reaction_files()
    
    # Try exact match first
    for name, filepath in available_files:
        if name.lower() == reaction_name.lower():
            with open(filepath, 'r') as f:
                return json.load(f), filepath
    
    # Try partial match
    for name, filepath in available_files:
        if reaction_name.lower() in name.lower() or name.lower() in reaction_name.lower():
            with open(filepath, 'r') as f:
                return json.load(f), filepath
    
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Enter your chemical-related question.")
    args = parser.parse_args()

    query = args.query_text.strip()
    if not query:
        print("Please enter a valid question.")
        return
    
    print(f"\n🔍 Searching for: {query}\n")
    
    # Check for specific reaction name mentions
    reaction_context = None
    source = None
    
    # Extract potential reaction names from the query
    potential_reaction_names = extract_reaction_names_from_query(query, threshold=80)
    
    if potential_reaction_names:
        print(f"Detected potential reaction names: {', '.join(potential_reaction_names)}")
        
        # Try each potential name
        for name in potential_reaction_names:
            reaction_data, reaction_file = find_reaction_file_by_name(name)
            if reaction_data:
                reaction_context = json.dumps(reaction_data, indent=2)
                source = os.path.basename(reaction_file)
                print(f"Found direct match in file: {source}")
                break

    # If no direct file match, use vector search
    if not reaction_context:
        print("Using vector similarity search...")
        # Initialize ChemBERTa and vector DB
        embeddings = ChemBERTaEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

        # Search for top results
        all_results = db.similarity_search_with_relevance_scores(query, k=5)
        
        print("\nTop 5 potential matches:")
        for i, (doc, score) in enumerate(all_results):
            try:
                content = json.loads(doc.page_content)
                reaction_name = content.get("reactionName", "Unknown")
                print(f"{i+1}. Score: {score:.4f}, Reaction: {reaction_name}")
                
                # Check if this document matches any of our potential reaction names
                if potential_reaction_names:
                    for name in potential_reaction_names:
                        if name.lower() in reaction_name.lower() or reaction_name.lower() in name.lower():
                            print(f"✓ Selected match {i+1} as it corresponds to the requested reaction type")
                            reaction_context = json.dumps(content, indent=2)
                            source = doc.metadata.get("source", "unknown")
                            break
                    if reaction_context:
                        break
            except json.JSONDecodeError:
                print(f"{i+1}. Score: {score:.4f}, Could not parse content")
        
        # If we still don't have a match, use the top result
        if not reaction_context:
            if not all_results or all_results[0][1] < 0.1:
                print("No relevant result found.")
                return
                
            top_doc, score = all_results[0]
            source = top_doc.metadata.get("source", "unknown")
            
            try:
                context_data = json.loads(top_doc.page_content)
                reaction_context = json.dumps(context_data, indent=2)
            except json.JSONDecodeError:
                reaction_context = top_doc.page_content  # fallback
    
    # Display the retrieved reaction data
    if reaction_context:
        print("\n🔍 Reaction Data:\n")
        print(reaction_context)

        print("\n🤖 AI Answer (via OpenAI):\n")
        answer = ask_openai(reaction_context, query)
        print(answer)

        print("\n📁 Source:", source)
    else:
        print("Could not retrieve any relevant reaction data.")


if __name__ == "__main__":
    main()