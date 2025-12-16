import os
import faiss
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

DOCS_DIR = "documents"
INDEX_DIR = "index"
os.makedirs(INDEX_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_documents():
    texts = []

    for file in os.listdir(DOCS_DIR):
        path = os.path.join(DOCS_DIR, file)

        if file.endswith(".txt") or file.endswith(".md"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())

        elif file.endswith(".pdf"):
            reader = PdfReader(path)
            content = ""
            for page in reader.pages:
                content += page.extract_text() or ""
            texts.append(content)

    return texts

def chunk_text(text, size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap

    return chunks

docs = load_documents()
chunks = []

for doc in docs:
    chunks.extend(chunk_text(doc))

embeddings = model.encode(chunks, show_progress_bar=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, f"{INDEX_DIR}/vectors.faiss")

with open(f"{INDEX_DIR}/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Index built successfully.")
