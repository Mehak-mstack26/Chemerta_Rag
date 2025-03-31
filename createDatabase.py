import os
import json
import glob
import shutil
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
from sklearn.preprocessing import normalize

load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "./data"

# ChemBERTa embedding model
class ChemBERTaEmbeddings:
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.model.eval()
        with torch.no_grad():
            embeddings = []
            for text in texts:
                tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                output = self.model(**tokens)
                cls_embedding = output.last_hidden_state[:, 0, :]  # CLS token
                np_embedding = cls_embedding.squeeze().numpy()
                norm_embedding = normalize([np_embedding])[0]  # L2 normalization
                embeddings.append(norm_embedding.tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

embeddings = ChemBERTaEmbeddings()

def load_documents():
    file_paths = glob.glob(os.path.join(DATA_PATH, "*.json"))
    documents = []

    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(Document(page_content=content, metadata={"source": path}))

    print(f"Loaded {len(documents)} files.")
    return documents

def split_json_documents(documents):
    json_documents = []
    for doc in documents:
        data = json.loads(doc.page_content)

        def process_dict(item):
            if isinstance(item, dict) and "allIncludedReactions" in item:
                for reaction in item["allIncludedReactions"]:
                    json_documents.append(Document(
                        page_content=json.dumps(reaction, indent=2),
                        metadata={"source": doc.metadata.get("source")}
                    ))
            else:
                json_documents.append(Document(
                    page_content=json.dumps(item, indent=2),
                    metadata={"source": doc.metadata.get("source")}
                ))

        if isinstance(data, list):
            for item in data:
                process_dict(item)
        elif isinstance(data, dict):
            process_dict(data)
    print(f"Split into {len(json_documents)} chunks.")
    return json_documents

def save_to_chroma(chunks: List[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to vector DB.")

def main():
    documents = load_documents()
    chunks = split_json_documents(documents)
    save_to_chroma(chunks)

if __name__ == "__main__":
    main()
