# Langgraph Agentic RAG

A conversational AI assistant built with LangGraph, featuring PDF-based Retrieval-Augmented Generation (RAG), web search, stock price lookup, and a calculator — all served through a Streamlit frontend. LangSmith is integrated for tracing, monitoring agent runs.

---

## Features

- **PDF RAG** — Upload a PDF and ask questions about its contents. Documents are chunked, embedded with OpenAI, and stored in a FAISS vector store per chat thread.
- **Web Search** — Uses DuckDuckGo to answer questions requiring up-to-date information.
- **Stock Price Lookup** — Fetches live stock quotes via Alpha Vantage.
- **Calculator** — Performs basic arithmetic (add, subtract, multiply, divide).
- **Multi-turn Memory** — Conversations are persisted per thread using SQLite checkpointing via LangGraph.
- **Multi-session Sidebar** — Browse and resume past conversations, similar to ChatGPT.
- **LangSmith Tracing** — Every agent run (LLM calls, tool invocations, chain steps) is traced automatically to LangSmith for debugging and observability.

---

## Architecture

```
streamlit_frontend.py        # UI layer (Streamlit)
langgraph_backend.py         # Agent graph, tools, PDF ingestion, state
chatbot.db                   # SQLite checkpoint store (auto-created)
```

The LangGraph graph follows a simple ReAct loop:

```
START → chat_node → (tool call?) → tools → chat_node → ...
```

Tools available to the agent: `rag_tool`, `search_tool`, `get_stock_price`, `calculator`.

---

## Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/account/api-keys)
- An [Alpha Vantage API key](https://www.alphavantage.co/support/#api-key) (free tier works)
- A [LangSmith API key](https://smith.langchain.com/) (free tier works)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/saznashaik/Langgraph_Agentic_RAG.git
cd Langgraph_Agentic_RAG
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=langgraph-agentic-rag   # any project name you like
```

The Alpha Vantage key is embedded in `langgraph_backend.py`. Replace it there if you want to use your own:

```python
# langgraph_backend.py, inside get_stock_price tool
url = (
    "https://www.alphavantage.co/query"
    f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=YOUR_KEY_HERE"
)
```

---

## Running the App

```bash
streamlit run streamlit_frontend.py
```

The app will open at `http://localhost:8501`.

---

## Usage

| Action | How |
|--------|-----|
| Upload a PDF | Attach a file in the chat input box |
| Ask about the PDF | Type any question after uploading |
| Search the web | Ask anything requiring current information |
| Get a stock price | "What is the current price of AAPL?" |
| Calculate | "What is 1234 multiplied by 56?" |
| New conversation | Click **New Chat** in the sidebar |
| Resume a conversation | Click any past chat title in the sidebar |

---

## Project Structure

```
Langgraph_Agentic_RAG/
├── langgraph_backend.py    # LLM, tools, graph definition, PDF ingestion
├── streamlit_frontend.py   # Streamlit UI, session state, streaming
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Dependencies (key packages)

| Package | Purpose |
|---------|---------|
| `langsmith` | Tracing, monitoring, and evaluation |
| `langgraph` | Agent graph orchestration |
| `langchain-openai` | LLM + embeddings |
| `langchain-community` | FAISS, DuckDuckGo, PDF loader |
| `streamlit` | Web UI |
| `openai` | OpenAI API client |
| `faiss-cpu` | Vector similarity search |

See `requirements.txt` for the full pinned list.

---

## LangSmith — Tracing 

This project uses [LangSmith](https://smith.langchain.com/) for observability and offline evaluation of the agent.

### Tracing

With `LANGCHAIN_TRACING_V2=true` set in your `.env`, every run is automatically traced. Each trace captures:

- the full message history sent to the LLM
- tool calls made (name, inputs, outputs)
- token usage and latency per step
- the final response

Traces appear in your LangSmith project dashboard in real time and are grouped by the `run_name` set in the Streamlit config (`"chat_turn"`).





MIT — see [LICENSE](LICENSE) for details.
