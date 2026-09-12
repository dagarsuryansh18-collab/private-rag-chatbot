"""
================================================================================
 PRIVATE RAG KNOWLEDGE REPOSITORY — STEP 2
 Cloud LLM Integration & RAG Chain (Groq + LCEL)
 Author: Principal RAG Architect
 Purpose: Connect the local Chroma retriever built in Step 1 to a fast,
          free-tier Groq-hosted LLM (openai/gpt-oss-20b) via a strict,
          context-grounded LCEL chain. The vector DB and embeddings stay
          100% local/private — only the final retrieved text snippets are
          sent to Groq's API for answer generation.

 DESIGN NOTES
 ------------
 - Modular: `get_rag_chain()` and `ask()` are self-contained and import-safe,
   so a later Streamlit UI (Step 3) can simply do:
       from rag_chain import ask
       answer = ask("some question")
 - No global chain is built at import time — the chain is constructed lazily
   inside `get_rag_chain()` so importing this module never fails just
   because the vector DB or API key isn't ready yet.
 - Strict grounding: the prompt explicitly forbids the model from using
   outside knowledge, and mandates a fixed fallback string when the
   retrieved context doesn't contain the answer.
================================================================================

INSTALLATION (run once, in a virtual environment — in addition to Step 1's deps)
----------------------------------------------------------------------------------
pip install langchain-groq
pip install python-dotenv   # optional, for loading GROQ_API_KEY from a .env file

SETUP
-----
Set your Groq API key as an environment variable before running:
    export GROQ_API_KEY="your-key-here"        (macOS/Linux)
    setx GROQ_API_KEY "your-key-here"           (Windows, new terminal after)

Or place it in a local ".env" file (never commit this file):
    GROQ_API_KEY=your-key-here
"""

import os
import logging
from typing import Optional, Dict, Any

# --- Local Step 1 backend (retriever) ---
try:
    from rag_backend import get_local_retriever
except ImportError as e:
    raise ImportError(
        "Could not import 'rag_backend.py' (Step 1). Make sure "
        "rag_backend.py is in the same directory/module path as this file. "
        f"Original error: {e}"
    )

# --- Groq LLM + LCEL primitives ---
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    from langchain_core.documents import Document
except ImportError as e:
    raise ImportError(
        "One or more required LangChain/Groq packages are missing.\n"
        "Please run:\n"
        "pip install langchain-groq langchain-core\n"
        f"Original error: {e}"
    )

# Optional: load a local .env file if python-dotenv is installed.
# This is best-effort only — never raises if dotenv is unavailable.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --------------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# --------------------------------------------------------------------------------
GROQ_MODEL_NAME: str = "openai/gpt-oss-20b"
LLM_TEMPERATURE: float = 0.1  # low temperature => factual, low-hallucination answers
FALLBACK_MESSAGE: str = "I cannot find this information in the repository."

STRICT_SYSTEM_PROMPT: str = """You are a private knowledge repository assistant.

Answer the user's question using ONLY the retrieved Context supplied in the user message.

RULES:
1. Use only information explicitly contained in Context.
2. Do not use outside knowledge or assumptions.
3. If Context does not contain enough information to answer the question, respond EXACTLY:
{fallback_message}
4. If Context does contain the answer, answer directly and concisely.
5. Do not say you cannot find the information if the Context clearly contains it.
"""

# --------------------------------------------------------------------------------
# LOGGING SETUP
# --------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("private_rag_chain")


# --------------------------------------------------------------------------------
# INTERNAL HELPER: SECURE API KEY LOADING
# --------------------------------------------------------------------------------
def _get_groq_api_key() -> Optional[str]:
    return os.getenv("GROQ_API_KEY")


# --------------------------------------------------------------------------------
# INTERNAL HELPER: LLM FACTORY
# --------------------------------------------------------------------------------
def _get_groq_llm() -> Optional[ChatGroq]:
    """
    Instantiate the Groq-hosted ChatGroq LLM client.

    Returns:
        Optional[ChatGroq]: A configured ChatGroq instance, or None if the
                             API key is missing or the client fails to build.
    """
    api_key = _get_groq_api_key()
    if not api_key:
        return None

    try:
        llm = ChatGroq(
            model=GROQ_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            groq_api_key=api_key,
        )
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize ChatGroq client with model "
                      f"'{GROQ_MODEL_NAME}'. Error: {e}")
        return None


# --------------------------------------------------------------------------------
# INTERNAL HELPER: CONTEXT FORMATTING
# --------------------------------------------------------------------------------
def _format_docs(docs: list) -> str:
    """
    Format a list of retrieved Document objects into a single context string
    suitable for insertion into the prompt template.

    Args:
        docs (list): List of langchain_core.documents.Document objects.

    Returns:
        str: A newline-separated, source-tagged context block. Returns an
             empty-context marker string if no documents were retrieved.
    """
    if not docs:
        return "(No relevant context was retrieved from the repository.)"

    formatted_chunks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown_source")
        page = doc.metadata.get("page", "N/A")
        formatted_chunks.append(
            f"[Chunk {i} | source: {source} | page: {page}]\n{doc.page_content.strip()}"
        )
    return "\n\n".join(formatted_chunks)


# --------------------------------------------------------------------------------
# STEP 2a: BUILD THE LCEL RAG CHAIN
# --------------------------------------------------------------------------------
def get_rag_chain(k: int = 3):
    """
    Construct and return the full LCEL RAG chain:
    retriever -> context formatting -> strict prompt -> Groq LLM -> string output.

    This function does NOT build the chain at import time — it is called
    lazily so importing this module never fails due to a missing vector DB
    or API key. Each call re-resolves the retriever and LLM, so it will
    pick up a freshly ingested PDF (Step 1) without needing an app restart.

    Args:
        k (int): Number of chunks to retrieve per query. Defaults to 3.

    Returns:
        Optional[Runnable]: A runnable LCEL chain accepting a plain string
                             question and returning a plain string answer,
                             or None if the retriever or LLM could not be
                             initialized (errors are logged).
    """
    # --- 1. Load retriever from Step 1 ---
    try:
        retriever = get_local_retriever(k=k)
    except Exception as e:
        logger.error(f"Unexpected error while loading retriever: {e}")
        return None

    if retriever is None:
        logger.error(
            "Retriever unavailable. Ensure a PDF has been ingested via "
            "ingest_pdf() before building the RAG chain."
        )
        return None

    # --- 2. Load Groq LLM ---
    llm = _get_groq_llm()
    if llm is None:
        logger.error("Groq LLM unavailable. Cannot build RAG chain.")
        return None

    # --- 3. Build strict prompt template ---
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", STRICT_SYSTEM_PROMPT.format(fallback_message=FALLBACK_MESSAGE)),
            ("human", "Retrieved Context:\n\n{context}\n\nUser Question:\n{question}"),
        ])
    except Exception as e:
        logger.error(f"Failed to build prompt template. Error: {e}")
        return None

    # --- 4. Assemble the LCEL chain ---
    try:
        rag_chain = (
            {
                "context": retriever | RunnableLambda(_format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain
    except Exception as e:
        logger.error(f"Failed to assemble LCEL RAG chain. Error: {e}")
        return None


# --------------------------------------------------------------------------------
# STEP 2b: SIMPLE QUERY ENTRY POINT (for easy Streamlit wiring)
# --------------------------------------------------------------------------------
def ask(question: str, k: int = 3) -> str:
    """
    High-level convenience function: builds the RAG chain (if needed) and
    answers a single question in one call. This is the primary function a
    Streamlit UI (Step 3) should import and call directly.

    Args:
        question (str): The user's natural-language question.
        k (int): Number of chunks to retrieve per query. Defaults to 3.

    Returns:
        str: The model's grounded answer, the strict fallback message if
             the repository doesn't contain the answer, or a clear
             human-readable error string if the chain could not run at all
             (e.g. missing API key, empty vector DB). This function never
             raises — it always returns a string, making it safe to bind
             directly to a UI button handler.
    """
    if not question or not isinstance(question, str) or not question.strip():
        return "Please enter a valid, non-empty question."

    try:
        chain = get_rag_chain(k=k)
    except Exception as e:
        logger.error(f"Unexpected error while building RAG chain: {e}")
        return ("An unexpected error occurred while setting up the RAG "
                 "pipeline. Check the logs for details.")

    if chain is None:
        return (
            "The RAG pipeline is not ready. Please verify: "
            "(1) at least one PDF has been ingested, and "
            "(2) the GROQ_API_KEY environment variable is set correctly."
        )

    try:
        answer = chain.invoke(question.strip())
        return answer if answer else FALLBACK_MESSAGE
    except Exception as e:
        logger.error(f"RAG chain invocation failed for question "
                      f"'{question}'. Error: {e}")
        return ("An error occurred while contacting the language model. "
                "This may be due to an invalid API key, network issue, or "
                "rate limiting. Please try again.")


# --------------------------------------------------------------------------------
# STANDALONE TEST / MANUAL RUN (safe to remove or wrap when plugging into Streamlit)
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print(" PRIVATE RAG CHAIN — STEP 2 — MANUAL TEST MODE")
    print("=" * 70)

    if not os.getenv("GROQ_API_KEY"):
        print(
            "\n⚠️  GROQ_API_KEY is not set in this environment.\n"
            "   Set it with: export GROQ_API_KEY=your-key-here\n"
        )
    else:
        print("\n✅ GROQ_API_KEY detected.\n")

    while True:
        user_question = input("Ask a question (or press Enter to quit): ").strip()
        if not user_question:
            print("Exiting test mode.")
            break
        response = ask(user_question)
        print(f"\n--- Answer ---\n{response}\n")