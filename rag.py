import faiss
import pickle
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

INDEX_DIR = "index"
MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

embedder = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(f"{INDEX_DIR}/vectors.faiss")
with open(f"{INDEX_DIR}/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8,
)

def retrieve(query, k=4):
    q_emb = embedder.encode([query])
    _, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]

def ask(query):
    context = "\n\n".join(retrieve(query))

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    output = llm(prompt, max_tokens=512, stop=["</s>"])
    print(output["choices"][0]["text"].strip())

if __name__ == "__main__":
    print("Local RAG ready. Type 'exit' to quit.\n")

    while True:
        q = input(">> ")
        if q.lower() == "exit":
            break
        ask(q)
