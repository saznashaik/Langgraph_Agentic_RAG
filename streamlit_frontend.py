import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)



# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])



def load_chat(thread_id):
    """Load a selected chat."""
    st.session_state["thread_id"] = thread_id

    messages = load_conversation(thread_id)

    temp_messages = []

    for msg in messages:

        role = (
            "user"
            if isinstance(msg, HumanMessage)
            else "assistant"
        )

        temp_messages.append(
            {
                "role": role,
                "content": msg.content
            }
        )

    st.session_state["message_history"] = temp_messages


def get_chat_title(thread_id):
    """
    Use the first user message as the title,
    similar to ChatGPT.
    """

    messages = load_conversation(thread_id)

    for msg in messages:
        if isinstance(msg, HumanMessage):

            title = msg.content.strip()

            if len(title) > 40:
                title = title[:40] + "..."

            return title

    return "New Chat"

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("🤖 Agentic AI")
st.sidebar.caption("PDF RAG + LangGraph")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    # newest chats first
    for thread_id in st.session_state['chat_threads'][::-1]:

        title = get_chat_title(thread_id)

        if st.sidebar.button(
            title,
            key=f"thread_{thread_id}",
            use_container_width=True
        ):
            load_chat(thread_id)
            st.rerun()


# ============================ Main Layout ========================
if not st.session_state["message_history"]:

    st.markdown("""
    <div class="chat-title">
        <h1>🤖 Agentic AI Assistant</h1>
        <p>Upload PDFs, ask questions, and use tools</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.title("🤖 Agentic AI Assistant")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Ask about your document...",
    accept_file=True
)

if prompt:
    user_input = prompt.text

    uploaded_files = prompt.files

    if uploaded_files:

        for uploaded_pdf in uploaded_files:

            if uploaded_pdf.name in thread_docs:
                st.info(f"📄 {uploaded_pdf.name} already indexed.")
                continue

            with st.status(f"Indexing {uploaded_pdf.name}...", expanded=True):

                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )

                thread_docs[uploaded_pdf.name] = summary
                st.session_state["active_document"] = uploaded_pdf.name
                st.success(f"✅ {uploaded_pdf.name} indexed")

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    # Build document-aware prompt
    message_content = user_input

    if "active_document" in st.session_state:

        active_doc = st.session_state["active_document"]

        message_content = f"""
    Current uploaded PDF: {active_doc}

    The user may refer to this document using phrases like:
    - this pdf
    - this document
    - summarize this
    - summarize this one
    - explain this
    - explain the document
    - what is in this file

    Assume the user is referring to the uploaded PDF unless they explicitly mention another document.

    User request:
    {user_input}
    """
        
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=message_content)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": msg.content})
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()