"""
================================================================================
 PRIVATE RAG KNOWLEDGE REPOSITORY — STEP 3
 Streamlit Web UI
 Author: Principal RAG Architect
 Purpose: Thin UI layer on top of the Step 1 (rag_backend.py) and Step 2
          (rag_chain.py) modules. This file contains NO ingestion or LLM
          logic of its own — it only wires user interactions (file upload,
          chat input) to the backend functions and renders the results.

 DESIGN NOTES
 ------------
 - Modular by construction: all heavy lifting (embeddings, Chroma, Groq)
   lives in rag_backend.py / rag_chain.py. This file is UI-only.
 - Uploaded PDFs are written to a local "./uploaded_pdfs" folder before
   being passed to ingest_pdf(pdf_path), since ingest_pdf() expects a
   filesystem path, not an in-memory buffer.
 - Chat history and per-message source metadata are kept in
   st.session_state so they survive Streamlit's rerun-on-interaction model.
 - Every backend call is wrapped defensively — a failure anywhere (missing
   API key, empty vector DB, bad PDF) surfaces as a clean st.error/st.warning
   message instead of a crash/traceback.
================================================================================

INSTALLATION (run once, in addition to Step 1 & Step 2 dependencies)
----------------------------------------------------------------------
pip install streamlit

RUN
---
streamlit run streamlit_app.py
"""

import os
import logging
import traceback

import streamlit as st

# --- Local Step 1 & Step 2 backend modules ---
try:
    from rag_backend import ingest_pdf, get_local_retriever, PERSIST_DIRECTORY
except ImportError as e:
    st.error(
        "Could not import 'rag_backend.py' (Step 1). Make sure it is in "
        f"the same directory as this file. Original error: {e}"
    )
    st.stop()

try:
    from rag_chain import ask, FALLBACK_MESSAGE
except ImportError as e:
    st.error(
        "Could not import 'rag_chain.py' (Step 2). Make sure it is in "
        f"the same directory as this file. Original error: {e}"
    )
    st.stop()

# --------------------------------------------------------------------------------
# CONFIGURATION CONSTANTS
# --------------------------------------------------------------------------------
UPLOAD_DIRECTORY: str = "./uploaded_pdfs"
APP_TITLE: str = "Private Knowledge Repository"
APP_ICON: str = "🔒"

# --------------------------------------------------------------------------------
# LOGGING SETUP
# --------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("private_rag_ui")


# --------------------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit call)
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------------------------------------
def _init_session_state() -> None:
    """
    Initialize all Streamlit session_state keys used by this app, if they
    don't already exist. Safe to call on every rerun.
    """
    if "chat_history" not in st.session_state:
        # Each entry: {"role": "user"/"assistant", "content": str, "sources": list}
        st.session_state.chat_history = []

    if "ingested_files" not in st.session_state:
        # Names of files successfully ingested in this session
        st.session_state.ingested_files = []

    if "ingestion_log" not in st.session_state:
        # (filename, status_message, is_success) tuples for sidebar display
        st.session_state.ingestion_log = []

    if "repository_ready" not in st.session_state:
        st.session_state.repository_ready = os.path.isdir(PERSIST_DIRECTORY)


# --------------------------------------------------------------------------------
# HELPER: SAVE UPLOADED FILE TO DISK
# --------------------------------------------------------------------------------
def _save_uploaded_file(uploaded_file) -> str:
    """
    Persist a Streamlit UploadedFile object to local disk so it can be
    passed as a path to ingest_pdf().

    Args:
        uploaded_file: A Streamlit UploadedFile object from st.file_uploader.

    Returns:
        str: The local filesystem path the file was saved to.

    Raises:
        OSError: If the upload directory cannot be created or the file
                 cannot be written.
    """
    try:
        os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
    except OSError as e:
        raise OSError(f"Could not create upload directory '{UPLOAD_DIRECTORY}'. "
                       f"Error: {e}")

    file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_file.name)
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except OSError as e:
        raise OSError(f"Could not write uploaded file to '{file_path}'. "
                       f"Error: {e}")

    return file_path


# --------------------------------------------------------------------------------
# HELPER: HANDLE A SINGLE PDF UPLOAD + INGESTION
# --------------------------------------------------------------------------------
def _handle_pdf_ingestion(uploaded_file) -> None:
    """
    Save an uploaded PDF to disk, ingest it via rag_backend.ingest_pdf(),
    and record the outcome in session_state for sidebar status display.

    Args:
        uploaded_file: A Streamlit UploadedFile object.
    """
    filename = uploaded_file.name

    if filename in st.session_state.ingested_files:
        st.sidebar.info(f"'{filename}' was already ingested this session.")
        return

    with st.sidebar.status(f"Processing '{filename}'...", expanded=True) as status:
        try:
            st.write("Saving file locally...")
            file_path = _save_uploaded_file(uploaded_file)

            st.write("Splitting text & generating local embeddings...")
            num_chunks = ingest_pdf(file_path)

            if num_chunks:
                st.session_state.ingested_files.append(filename)
                st.session_state.ingestion_log.append(
                    (filename, f"{num_chunks} chunks ingested", True)
                )
                st.session_state.repository_ready = True
                status.update(
                    label=f"'{filename}' ingested ({num_chunks} chunks)",
                    state="complete",
                )
            else:
                st.session_state.ingestion_log.append(
                    (filename, "Ingestion failed — see logs", False)
                )
                status.update(label=f"Failed to ingest '{filename}'", state="error")

        except OSError as e:
            st.session_state.ingestion_log.append((filename, str(e), False))
            status.update(label=f"File error for '{filename}'", state="error")
        except Exception as e:
            logger.error(f"Unexpected error ingesting '{filename}': {e}\n"
                          f"{traceback.format_exc()}")
            st.session_state.ingestion_log.append(
                (filename, f"Unexpected error: {e}", False)
            )
            status.update(label=f"Unexpected error for '{filename}'", state="error")


# --------------------------------------------------------------------------------
# SIDEBAR: PDF UPLOAD + INGESTION STATUS
# --------------------------------------------------------------------------------
def render_sidebar() -> None:
    """
    Render the sidebar: PDF uploader, per-file ingestion status log, and an
    overall repository readiness indicator.
    """
    st.sidebar.title(f"{APP_ICON} Knowledge Ingestion")
    st.sidebar.caption(
        "Upload PDFs to add them to your private, local knowledge base. "
        "Everything is processed and stored on this machine only."
    )

    uploaded_files = st.sidebar.file_uploader(
        label="Upload PDF document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can select multiple PDFs at once. Each will be split, "
             "embedded locally, and stored in the private vector database.",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            _handle_pdf_ingestion(uploaded_file)

    st.sidebar.divider()

    # --- Overall repository status ---
    if st.session_state.repository_ready:
        st.sidebar.success("✅ Vector database is ready for queries.")
    else:
        st.sidebar.warning("⚠️ No documents ingested yet. Upload a PDF to begin.")

    # --- Per-file ingestion log ---
    if st.session_state.ingestion_log:
        st.sidebar.subheader("Ingestion Log")
        for filename, message, is_success in reversed(st.session_state.ingestion_log):
            icon = "✅" if is_success else "❌"
            st.sidebar.markdown(f"{icon} **{filename}** — {message}")

    st.sidebar.divider()

    # --- Environment / config diagnostics ---
    with st.sidebar.expander("⚙️ System Status", expanded=False):
        groq_key_set = bool(os.getenv("GROQ_API_KEY"))
        st.markdown(
            f"- GROQ_API_KEY: {'✅ set' if groq_key_set else '❌ not set'}\n"
            f"- Vector DB path: `{PERSIST_DIRECTORY}`\n"
            f"- Files ingested this session: {len(st.session_state.ingested_files)}"
        )
        if not groq_key_set:
            st.caption(
                "Set GROQ_API_KEY as an environment variable before asking "
                "questions, e.g. `export GROQ_API_KEY=your-key-here`."
            )

    if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# --------------------------------------------------------------------------------
# HELPER: RETRIEVE SOURCE CHUNKS FOR DISPLAY (separate from the answer call)
# --------------------------------------------------------------------------------
def _get_source_chunks(question: str, k: int = 3) -> list:
    """
    Re-run retrieval (independent of the LLM call) purely to surface source
    metadata (file name + page number) for the "View Sources" expander.

    This is a read-only, local-only operation (no Groq API call), so it's
    safe to run even if the LLM call itself fails.

    Args:
        question (str): The user's question.
        k (int): Number of chunks to retrieve.

    Returns:
        list: A list of dicts with keys 'source_file', 'page', 'snippet'.
              Empty list if retrieval is unavailable or fails.
    """
    try:
        retriever = get_local_retriever(k=k)
        if retriever is None:
            return []
        docs = retriever.invoke(question)
        sources = []
        for doc in docs:
            sources.append({
                "source_file": doc.metadata.get("source_file", "unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "snippet": doc.page_content.strip()[:400],
            })
        return sources
    except Exception as e:
        logger.warning(f"Could not retrieve source chunks for display: {e}")
        return []


# --------------------------------------------------------------------------------
# CHAT INTERFACE
# --------------------------------------------------------------------------------
def render_chat_interface() -> None:
    """
    Render the main chat interface: message history, chat input box, and
    per-answer expandable source citations.
    """
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption(
        "Ask questions about your ingested documents. Answers are grounded "
        "strictly in your private repository — nothing is invented."
    )

    # --- Render existing chat history ---
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📄 View Retrieved Sources"):
                    for i, src in enumerate(message["sources"], start=1):
                        st.markdown(
                            f"**Source {i}:** `{src['source_file']}` "
                            f"— page {src['page']}"
                        )
                        st.caption(src["snippet"])
                        st.divider()

    # --- Chat input ---
    user_question = st.chat_input("Ask a question about your documents...")

    if user_question:
        # Guard: no documents ingested yet
        if not st.session_state.repository_ready:
            with st.chat_message("assistant"):
                st.warning(
                    "No documents have been ingested yet. Please upload a "
                    "PDF in the sidebar before asking questions."
                )
            return

        # Guard: no Groq API key
        if not os.getenv("GROQ_API_KEY"):
            with st.chat_message("assistant"):
                st.error(
                    "GROQ_API_KEY is not set. Please set it as an "
                    "environment variable and restart the app."
                )
            return

        # Display the user's message immediately
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question, "sources": []}
        )
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate and display the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Searching the repository and generating an answer..."):
                try:
                    answer = ask(user_question)
                except Exception as e:
                    logger.error(f"Unexpected error during ask(): {e}\n"
                                  f"{traceback.format_exc()}")
                    answer = (
                        "An unexpected error occurred while generating the "
                        "answer. Please check the application logs."
                    )

                sources = []
                if answer and answer.strip() != FALLBACK_MESSAGE.strip():
                    sources = _get_source_chunks(user_question)

            st.markdown(answer)

            if sources:
                with st.expander("📄 View Retrieved Sources"):
                    for i, src in enumerate(sources, start=1):
                        st.markdown(
                            f"**Source {i}:** `{src['source_file']}` "
                            f"— page {src['page']}"
                        )
                        st.caption(src["snippet"])
                        st.divider()

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


# --------------------------------------------------------------------------------
# MAIN APP ENTRY POINT
# --------------------------------------------------------------------------------
def main() -> None:
    """
    Application entry point. Initializes session state and renders the
    sidebar and chat interface. Wrapped in a top-level try/except so any
    unforeseen error produces a clean Streamlit error message instead of
    an unhandled traceback in the browser.
    """
    try:
        _init_session_state()
        render_sidebar()
        render_chat_interface()
    except Exception as e:
        logger.error(f"Fatal error in main app loop: {e}\n{traceback.format_exc()}")
        st.error(
            "A critical error occurred while running the application. "
            "Please check the terminal logs for full details."
        )


if __name__ == "__main__":
    main()