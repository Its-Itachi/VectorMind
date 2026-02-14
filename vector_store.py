from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


# ------------------------------------------------------------
# Initialize Embedding Model (loaded once)
# ------------------------------------------------------------
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ------------------------------------------------------------
# BUILD FAISS INDEX
# ------------------------------------------------------------
def build_faiss(chunks, index_path="vector_store"):
    """
    Create and save a FAISS store from normalized chunks.
    """

    texts = []
    metadatas = []

    for ch in chunks:
        if not ch:
            continue

        # Prefer embedding_text
        content = ch.get("embedding_text") or ch.get("content") or ""

        if not content.strip():
            continue

        texts.append(content)

        metadatas.append({
            "page": ch.get("page"),
            "type": ch.get("type"),
            "raw": ch.get("content"),
        })

    # ------------------------------------------------------------
    # SAFETY CHECK — Prevent FAISS crash
    # ------------------------------------------------------------
    if not texts:
        print("\nNo valid chunks found. FAISS index NOT created.")
        return None

    print(f"\nBuilding FAISS with {len(texts)} vectors...")

    # Create FAISS store
    store = FAISS.from_texts(
        texts=texts,
        embedding=emb,
        metadatas=metadatas
    )

    # Save locally
    store.save_local(index_path)

    print(f"FAISS index built & saved to '{index_path}'")
    print(f"Total vectors stored: {len(texts)}")

    return store


# ------------------------------------------------------------
# LOAD FAISS INDEX
# ------------------------------------------------------------
def load_faiss(index_path="vector_store"):
    """
    Load existing FAISS index.
    """

    if not os.path.exists(index_path):
        raise RuntimeError(f"FAISS index not found at '{index_path}'")

    print(f"Loading FAISS index from '{index_path}'...")

    store = FAISS.load_local(
        index_path,
        emb,
        allow_dangerous_deserialization=True
    )

    print("FAISS index loaded successfully.")
    return store
