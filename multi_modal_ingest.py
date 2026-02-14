from unstructured.partition.pdf import partition_pdf
from bs4 import BeautifulSoup
import pytesseract
import tempfile


# ---------------------------------------------------------------------------------------------------
# PDF → Unstructured Elements → Normalize Elements → Multi-Modal Chunks → Embeddings → FAISS Retriever
# ---------------------------------------------------------------------------------------------------


def multi_modal_ingest(file_bytes: bytes):
    """
    Ingest a PDF using Unstructured with AUTO strategy.
    AUTO handles both text and scanned PDFs without forcing OCR.
    """

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Use AUTO strategy (most stable)
    print("Using AUTO extraction strategy...")
    elements = partition_pdf(
        filename=tmp_path,
        strategy="auto",
        include_page_breaks=False,
    )

    print("TOTAL ELEMENTS:", len(elements))

    structured_output = []

    for el in elements:
        structured_output.append({
            "type": el.__class__.__name__,
            "content": getattr(el, "text", "") or "",
            "page": getattr(el.metadata, "page_number", None),
            "filetype": getattr(el.metadata, "filetype", None),
            "image_path": getattr(el, "image_path", None),
            "metadata": el.metadata.to_dict() if hasattr(el.metadata, "to_dict") else {}
        })

    return structured_output


def normalize_element(el, pdf_bytes):
    """
    Convert Unstructured element into unified RAG chunk.
    """

    element_type = el.get("type")
    page = el.get("page")
    text = el.get("content", "").strip()
    metadata = el.get("metadata", {})
    html = metadata.get("text_as_html")
    img_path = el.get("image_path")

    # -------------------------------------
    # CASE 1: NORMAL TEXT
    # -------------------------------------
    if element_type not in ["Table", "Image", "Picture"]:
        if not text:
            return None

        return {
            "type": "text",
            "page": page,
            "content": text,
            "image_path": None,
            "embedding_text": f"Text: {text}"
        }

    # -------------------------------------
    # CASE 2: TABLE
    # -------------------------------------
    if element_type == "Table":

        # HTML table handling
        if html:
            soup = BeautifulSoup(html, "html.parser")
            rows = soup.find_all("tr")

            markdown = []
            for r in rows:
                cols = [c.get_text(strip=True) for c in r.find_all(["td", "th"])]
                if len(cols) > 1:
                    markdown.append("| " + " | ".join(cols) + " |")

            if markdown:
                table_md = "\n".join(markdown)

                return {
                    "type": "table",
                    "page": page,
                    "content": table_md,
                    "image_path": None,
                    "embedding_text": f"Table: {table_md}",
                }

        # fallback as text
        if text:
            return {
                "type": "text",
                "page": page,
                "content": text,
                "image_path": None,
                "embedding_text": f"Text: {text}",
            }

    # -------------------------------------
    # CASE 3: IMAGE / PICTURE (Optional OCR)
    # -------------------------------------
    if element_type in ["Image", "Picture"] and img_path:
        try:
            ocr_text = pytesseract.image_to_string(img_path).strip()
        except Exception:
            ocr_text = ""

        if ocr_text:
            return {
                "type": "ocr",
                "page": page,
                "content": ocr_text,
                "image_path": img_path,
                "embedding_text": f"OCR: {ocr_text}",
            }

    return None
