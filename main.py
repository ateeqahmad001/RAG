from __future__ import annotations

import os

import streamlit as st

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import AVAILABLE_MODELS, DEFAULT_MODEL
from flow import run_pipeline

MAX_CONVERSATION_TURNS = 3


def initialize_session_state() -> None:
    defaults = {
        "messages": [],
        "groq_api_key": os.environ.get("GROQ_API_KEY", ""),
        "selected_model": DEFAULT_MODEL,
        "uploaded_pdf_bytes": None,
        "pdf_names": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_conversation_context() -> str:
    recent_turns = st.session_state.messages[-(MAX_CONVERSATION_TURNS * 2):]
    if not recent_turns:
        return ""

    context_lines = []
    for message in recent_turns:
        role_label = "User" if message["role"] == "user" else "Assistant"
        context_lines.append(f"{role_label}: {message['content']}")

    return "Previous conversation:\n" + "\n".join(context_lines)


def render_sidebar() -> None:
    with st.sidebar:
        st.title("🧠 Agentic RAG")
        st.caption("Powered by LangGraph + Groq")

        st.divider()

        st.subheader("🔑 API Configuration")
        groq_key_input = st.text_input(
            "Groq API Key",
            value=st.session_state.groq_api_key,
            type="password",
            help="Get your key at console.groq.com",
        )
        if groq_key_input:
            st.session_state.groq_api_key = groq_key_input

        st.divider()

        st.subheader("🤖 Model")
        selected_model = st.selectbox(
            "Choose LLM",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state.selected_model)
            if st.session_state.selected_model in AVAILABLE_MODELS
            else 0,
        )
        st.session_state.selected_model = selected_model

        st.divider()

        st.subheader("📄 Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDF files to query against.",
        )

        if uploaded_files:
            pdf_data = tuple(
                (uploaded_file.name, uploaded_file.read())
                for uploaded_file in uploaded_files
            )
            st.session_state.uploaded_pdf_bytes = pdf_data
            st.session_state.pdf_names = [f.name for f in uploaded_files]

        if st.session_state.pdf_names:
            st.success(f"Loaded: {', '.join(st.session_state.pdf_names)}")

        st.divider()

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        st.caption("Built with LangChain · LangGraph · Groq · FAISS · Streamlit")


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]

            if metadata.get("visited_nodes"):
                node_labels = {
                    "guardrail": "🛡️ Guardrail Check",
                    "isformal_check": "💬 Formality Check",
                    "formal_responder": "🗣️ Conversational Responder",
                    "retrieve": "📂 Document Retrieval",
                    "grade_document": "📊 Document Grading",
                    "increment_attempts": "🔁 Attempt Counter",
                    "web_search": "🌐 Web Search",
                    "generate_response": "✍️ Response Generation",
                    "fallback": "⚠️ Fallback Handler",
                }
                pipeline_steps = " → ".join(
                    node_labels.get(node, node)
                    for node in metadata["visited_nodes"]
                )
                with st.expander("🔍 Pipeline Trace", expanded=False):
                    st.markdown(f"**Path taken:** {pipeline_steps}")

            if metadata.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for idx, source in enumerate(metadata["sources"], 1):
                        pdf_name = source.get("pdf_name", "Unknown PDF")
                        page_num = source.get("page", "N/A")
                        snippet = source.get("snippet", "")[:300]
                        st.markdown(f"**Source {idx}** — `{pdf_name}` · Page {page_num}")
                        st.caption(snippet + ("..." if len(snippet) == 300 else ""))
                        if idx < len(metadata["sources"]):
                            st.divider()

def extract_source_metadata(documents: list) -> list[dict]:
    sources = []
    for doc in documents:
        if hasattr(doc, "page_content"):
            sources.append(
                {
                    "pdf_name": doc.metadata.get("pdf_name", "Unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "snippet": doc.page_content,
                }
            )
    return sources


def handle_user_query(user_query: str) -> None:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

    if not st.session_state.groq_api_key:
        with st.chat_message("assistant"):
            st.error("Please enter your Groq API key in the sidebar to continue.")
        return

    if not st.session_state.uploaded_pdf_bytes:
        with st.chat_message("assistant"):
            st.error("Please upload at least one PDF document in the sidebar.")
        return

    response_text = ""
    message_metadata = {"visited_nodes": [], "sources": []}

    with st.chat_message("assistant"):
        with st.status("Running agentic pipeline...", expanded=True) as pipeline_status:
            st.write("Checking query safety...")

            try:
                conversation_context = build_conversation_context()

                final_state = run_pipeline(
                    query=user_query,
                    pdf_file_bytes_list=st.session_state.uploaded_pdf_bytes,
                    groq_api_key=st.session_state.groq_api_key,
                    model_name=st.session_state.selected_model,
                    conversation_context=conversation_context,
                )

                visited_nodes = final_state.get("visited_nodes", [])
                node_display_names = {
                    "guardrail": "✅ Guardrail passed",
                    "isformal_check": "✅ Query classified",
                    "retrieve": "📂 Documents retrieved",
                    "grade_document": "📊 Documents graded",
                    "web_search": "🌐 Web search performed",
                    "generate_response": "✍️ Response generated",
                    "fallback": "⚠️ Fallback triggered",
                }
                for node in visited_nodes:
                    if node in node_display_names:
                        st.write(node_display_names[node])

                pipeline_status.update(label="Pipeline complete!", state="complete")

                response_text = final_state.get("generation", "") or "I was unable to generate a response. Please try again."
                documents = final_state.get("documents", [])
                sources = extract_source_metadata(documents)
                message_metadata = {"visited_nodes": visited_nodes, "sources": sources}

            except Exception as pipeline_error:
                pipeline_status.update(label="Pipeline error", state="error")
                st.error(f"An error occurred while processing your query: {str(pipeline_error)}")
                response_text = "I encountered an error processing your request. Please check your API keys and try again."

        st.markdown(response_text)

    node_labels = {
        "guardrail": "🛡️ Guardrail",
        "isformal_check": "💬 Formality",
        "formal_responder": "🗣️ Conversational",
        "retrieve": "📂 Retrieve",
        "grade_document": "📊 Grade",
        "increment_attempts": "🔁 Retry",
        "web_search": "🌐 Web Search",
        "generate_response": "✍️ Generate",
        "fallback": "⚠️ Fallback",
    }

    if message_metadata.get("visited_nodes"):
        pipeline_steps = " → ".join(
            node_labels.get(node, node) for node in message_metadata["visited_nodes"]
        )
        with st.expander("🔍 Pipeline Trace", expanded=False):
            st.markdown(f"**Path taken:** {pipeline_steps}")

    if message_metadata.get("sources"):
        with st.expander("📚 Sources", expanded=False):
            for idx, source in enumerate(message_metadata["sources"], 1):
                pdf_name = source.get("pdf_name", "Unknown PDF")
                page_num = source.get("page", "N/A")
                snippet = source.get("snippet", "")[:300]
                st.markdown(f"**Source {idx}** — `{pdf_name}` · Page {page_num}")
                st.caption(snippet + ("..." if len(snippet) == 300 else ""))
                if idx < len(message_metadata["sources"]):
                    st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "metadata": message_metadata,
        }
    )


def render_welcome_screen() -> None:
    st.title("🧠Agentic RAG")
    st.markdown(
        """
        Welcome to **DocMind**, an intelligent document question-answering system powered by an
        agentic retrieval-augmented generation (RAG) pipeline.

        ### Getting Started
        1. **Enter your Groq API key** in the sidebar
        2. **Upload one or more PDF documents** using the sidebar uploader
        3. **Ask questions** about your documents in the chat below

        ### Features
        - 🛡️ **Guardrail protection** : Filters harmful or inappropriate queries
        - 📂 **Smart retrieval** : FAISS-powered semantic search across all your PDFs
        - 🌐 **Web search fallback** : Searches the web when documents don't have the answer
        - 📊 **Relevance grading** : Filters out irrelevant retrieved chunks
        - 🔍 **Hallucination detection** : Verifies responses are grounded in facts
        - 💬 **Conversation memory** : Maintains context across follow-up questions
        - 📚 **Source attribution** : Shows exactly which document each answer came from
        """
    )
    st.divider()


def main() -> None:
    initialize_session_state()
    render_sidebar()

    if not st.session_state.messages:
        render_welcome_screen()

    render_chat_history()

    user_query = st.chat_input("Ask a question about your documents...")
    if user_query:
        handle_user_query(user_query)


if __name__ == "__main__":
    main()
