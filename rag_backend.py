"""
================================================================================
 PRIVATE RAG KNOWLEDGE REPOSITORY — STEP 1
 Local, private, zero-cost RAG ingestion + retrieval backend.
 Author: Principal RAG Architect
 Purpose: Ingest custom PDFs into a local Chroma vector DB using local
          HuggingFace embeddings, and expose a retriever for downstream
          use (e.g. a Streamlit UI in a later step).

 DESIGN NOTES
 ------------
 - 100% local & private: no API keys, no network calls, no external
   vector DB service. Everything runs on CPU.
 - Modular: `ingest_pdf()` and `get_local_retriever()` are independent,
   import-safe functions with no global state and no side effects on
   import (safe to `import rag_backend` from a Streamlit app).
 - Low-spec friendly: small embedding model (all-MiniLM-L6-v2, ~80MB),
   small chunk size, batched embedding via Chroma's internal handling,
   and no in-memory duplication of the whole document set.
================================================================================

INSTALLATION (run once, in a virtual environment)
--------------------------------------------------
pip install langchain
pip install langchain-community
pip install langchain-huggingface
pip install langchain-chroma
pip install chromadb
pip install pypdf
pip install sentence-transformers

(All-in-one line, if preferred:)
pip install langchain langchain-community langchain-huggingface langchain-chroma chromadb pypdf sentence-transformers
"""

import os
import logging
from typing import Optional, List

# --- LangChain / Chroma / HuggingFace imports (current, non-deprecated paths) ---
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStoreRetriever
except ImportError as e:
    raise ImportError(
        "One or more required LangChain packages are missing.\n"
        "Please run:\n"
        "pip install langchain langchain-community langchain-huggingface "
        "langchain-chroma chromadb pypdf sentence-transformers\n"
        f"Original error: {e}"
    )

# --------------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# --------------------------------------------------------------------------------
PERSIST_DIRECTORY: str = "./private_vector_db"
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
TOP_K: int = 3
COLLECTION_NAME: str = "private_knowledge_base"

# --------------------------------------------------------------------------------
# LOGGING SETUP
# --------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("private_rag_backend")


# --------------------------------------------------------------------------------
# INTERNAL HELPER: EMBEDDING MODEL FACTORY
# --------------------------------------------------------------------------------
def _get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Instantiate the local, CPU-based HuggingFace embedding model.

    Kept as a private helper so both `ingest_pdf()` and
    `get_local_retriever()` construct the embedding function identically,
    which is required for Chroma to correctly read/write the same
    vector space.

    Returns:
        HuggingFaceEmbeddings: A local embedding function object.

    Raises:
        RuntimeError: If the embedding model fails to load (e.g. missing
                      sentence-transformers dependency, corrupted cache).
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return embeddings
    except Exception as e:
        raise RuntimeError(
            f"Failed to load local embedding model '{EMBEDDING_MODEL_NAME}'. "
            f"Ensure 'sentence-transformers' is installed and you have "
            f"network access on first run to download model weights. "
            f"Original error: {e}"
        )


# --------------------------------------------------------------------------------
# INTERNAL HELPER: DIRECTORY SAFETY
# --------------------------------------------------------------------------------
def _ensure_persist_directory(path: str) -> None:
    """
    Ensure the vector DB persistence directory exists, creating it if needed.

    Args:
        path (str): Directory path to check/create.

    Raises:
        OSError: If the directory cannot be created due to permissions or
                 invalid path issues.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        raise OSError(
            f"Could not create persistence directory '{path}'. "
            f"Check folder permissions or disk space. Original error: {e}"
        )


# --------------------------------------------------------------------------------
# STEP 1a: PDF INGESTION PIPELINE
# --------------------------------------------------------------------------------
def ingest_pdf(pdf_path: str) -> Optional[int]:
    """
    Load a PDF, split it into overlapping chunks, generate local embeddings,
    and persist them into the local Chroma vector database.

    This function is fully self-contained and idempotent-safe: calling it
    multiple times with different PDFs will keep adding documents to the
    same persistent collection at `PERSIST_DIRECTORY`.

    Args:
        pdf_path (str): Path to the PDF file to ingest.

    Returns:
        Optional[int]: Number of chunks successfully ingested, or None if
                        ingestion failed at any stage (errors are logged,
                        not raised, so this is safe to call from a UI
                        event handler without crashing the app).

    Workflow:
        1. Validate the file path exists and is a .pdf file.
        2. Load the PDF using PyPDFLoader (page-wise Document objects).
        3. Split documents into chunks via RecursiveCharacterTextSplitter.
        4. Generate local embeddings via HuggingFace MiniLM model.
        5. Persist chunks + embeddings into the local Chroma DB.
    """
    # --- 1. Validate input path ---
    if not pdf_path or not isinstance(pdf_path, str):
        logger.error("Invalid pdf_path argument: must be a non-empty string.")
        return None

    if not os.path.isfile(pdf_path):
        logger.error(f"PDF file not found at path: '{pdf_path}'")
        return None

    if not pdf_path.lower().endswith(".pdf"):
        logger.error(f"File '{pdf_path}' does not have a .pdf extension.")
        return None

    # --- 2. Load the PDF ---
    try:
        loader = PyPDFLoader(pdf_path)
        raw_documents: List[Document] = loader.load()
    except Exception as e:
        logger.error(f"Failed to load/parse PDF '{pdf_path}'. It may be "
                      f"corrupted, encrypted, or not a valid PDF. Error: {e}")
        return None

    if not raw_documents:
        logger.warning(f"No extractable text found in '{pdf_path}'. "
                        f"It may be a scanned/image-only PDF requiring OCR.")
        return None

    # --- 3. Split into chunks ---
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks: List[Document] = text_splitter.split_documents(raw_documents)
    except Exception as e:
        logger.error(f"Text splitting failed for '{pdf_path}'. Error: {e}")
        return None

    if not chunks:
        logger.warning(f"Splitting produced zero chunks for '{pdf_path}'.")
        return None

    # Tag each chunk with its source filename for traceability in the UI
    source_name = os.path.basename(pdf_path)
    for chunk in chunks:
        chunk.metadata["source_file"] = source_name

    # --- 4. Prepare embedding model ---
    try:
        embeddings = _get_embedding_model()
    except RuntimeError as e:
        logger.error(str(e))
        return None

    # --- 5. Ensure persistence directory exists ---
    try:
        _ensure_persist_directory(PERSIST_DIRECTORY)
    except OSError as e:
        logger.error(str(e))
        return None

    # --- 6. Embed and persist into Chroma ---
    try:
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )
        vector_store.add_documents(documents=chunks)
    except Exception as e:
        logger.error(f"Failed to embed/persist chunks into Chroma DB. "
                      f"Check disk space and write permissions on "
                      f"'{PERSIST_DIRECTORY}'. Error: {e}")
        return None

    logger.info(f"Successfully ingested '{source_name}': "
                f"{len(chunks)} chunks added to '{PERSIST_DIRECTORY}'.")
    return len(chunks)


# --------------------------------------------------------------------------------
# STEP 1b: RETRIEVER LOADER
# --------------------------------------------------------------------------------
def get_local_retriever(k: int = TOP_K) -> Optional[VectorStoreRetriever]:
    """
    Load the persisted local Chroma vector database and return a retriever
    instance ready for similarity search (e.g. inside a RAG chain or a
    Streamlit query handler).

    Args:
        k (int): Number of top similar chunks to retrieve. Defaults to
                  module-level TOP_K (3).

    Returns:
        Optional[VectorStoreRetriever]: A configured retriever, or None if
                                         the vector DB doesn't exist yet or
                                         fails to load (errors are logged).

    Usage (for later Streamlit integration):
        retriever = get_local_retriever()
        if retriever:
            relevant_docs = retriever.invoke("your question here")
    """
    # --- 1. Check the DB directory actually exists ---
    if not os.path.isdir(PERSIST_DIRECTORY):
        logger.error(
            f"No vector database found at '{PERSIST_DIRECTORY}'. "
            f"Run ingest_pdf() on at least one document first."
        )
        return None

    # --- 2. Load embedding model (must match ingestion-time model) ---
    try:
        embeddings = _get_embedding_model()
    except RuntimeError as e:
        logger.error(str(e))
        return None

    # --- 3. Load the persisted Chroma collection ---
    try:
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIRECTORY,
        )
    except Exception as e:
        logger.error(f"Failed to load Chroma DB from '{PERSIST_DIRECTORY}'. "
                      f"The DB folder may be corrupted or empty. Error: {e}")
        return None

    # --- 4. Sanity check: does the collection actually contain vectors? ---
    try:
        doc_count = vector_store._collection.count()
        if doc_count == 0:
            logger.warning(
                "Vector DB exists but contains 0 documents. "
                "Ingest a PDF before querying."
            )
            return None
    except Exception as e:
        logger.warning(f"Could not verify document count in vector DB: {e}")
        # Non-fatal — proceed to return retriever anyway.

    # --- 5. Build and return retriever ---
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )
        return retriever
    except Exception as e:
        logger.error(f"Failed to construct retriever from vector store. "
                      f"Error: {e}")
        return None


# --------------------------------------------------------------------------------
# STANDALONE TEST / MANUAL RUN (safe to remove or wrap when plugging into Streamlit)
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print(" PRIVATE RAG BACKEND — STEP 1 — MANUAL TEST MODE")
    print("=" * 70)

    test_pdf_path = input(
        "Enter path to a PDF file to ingest (or press Enter to skip): "
    ).strip()

    if test_pdf_path:
        num_chunks = ingest_pdf(test_pdf_path)
        if num_chunks:
            print(f"\n✅ Ingested {num_chunks} chunks successfully.\n")
        else:
            print("\n❌ Ingestion failed. Check the logs above for details.\n")

    retriever = get_local_retriever()
    if retriever:
        print("✅ Retriever loaded successfully.")
        test_query = input("\nEnter a test query (or press Enter to exit): ").strip()
        if test_query:
            try:
                results = retriever.invoke(test_query)
                print(f"\nTop {len(results)} results:\n")
                for i, doc in enumerate(results, start=1):
                    source = doc.metadata.get("source_file", "unknown")
                    page = doc.metadata.get("page", "N/A")
                    print(f"--- Result {i} (source: {source}, page: {page}) ---")
                    print(doc.page_content[:300].strip())
                    print()
            except Exception as e:
                print(f"❌ Query failed: {e}")
    else:
        print("❌ No retriever available. Ingest a PDF ") 