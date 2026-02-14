from multi_modal_ingest import multi_modal_ingest, normalize_element
from vector_store import build_faiss

import os


PDF_PATH = "multi-modal_rag_qa_assignment.pdf"


def load_pdf_bytes(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found at: {path}")

    with open(path, "rb") as f:
        data = f.read()

    if not data:
        raise RuntimeError("PDF file loaded but is empty (0 bytes).")

    print(f"PDF size: {len(data)} bytes")
    return data


def main():

    print("\n--- LOADING PDF ---")
    pdf_bytes = load_pdf_bytes(PDF_PATH)

    print("\n--- EXTRACTING ELEMENTS ---")
    elements = multi_modal_ingest(pdf_bytes)

    if not elements:
        print("\nNo elements extracted from PDF.")
        print("Possible causes:")
        print("- Unsupported PDF structure")
        print("- Corrupted file")
        print("- Unstructured extraction issue")
        print("\nStopping pipeline.\n")
        return

    print(f"TOTAL ELEMENTS: {len(elements)}")

    chunks = []

    for el in elements:
        ch = normalize_element(el, pdf_bytes)

        if ch and ch.get("embedding_text"):
            chunks.append(ch)

    if not chunks:
        print("\nNo valid chunks after normalization.")
        print("Stopping before FAISS build.\n")
        return

    print(f"VALID CHUNKS: {len(chunks)}")

    print("\n--- BUILDING VECTOR STORE ---")
    build_faiss(chunks, "vector_store_mm")

    print("\nDONE\n")


if __name__ == "__main__":
    main()
