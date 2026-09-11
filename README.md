A hands-on educational Python project teaching how to connect real open-source **Model Context Protocol (MCP)** servers to **LangChain/LangGraph ReAct agents** using `langchain-mcp-adapters` and `MultiServerMCPClient`.

---

Keep this foundational distinction in mind throughout the project:

```
┌─────────────────────────────────────────────────────────────┐
│  MCP Tool    = Capability                                   │
│  ReAct Agent = Decision Maker                               │
│                                                             │
│  MCP         = Standardized protocol for communicating with │
│                tools and services.                          │
│  ReAct       = Agent pattern that enables an LLM to decide  │
│                WHEN and WHICH tool to invoke.               │
└─────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [What is MCP?](#1-what-is-mcp)
2. [What is an MCP Server?](#2-what-is-an-mcp-server)
3. [What is an MCP Client?](#3-what-is-an-mcp-client)
4. [What is an MCP Tool?](#4-what-is-an-mcp-tool)
5. [What is langchain-mcp-adapters?](#5-what-is-langchain-mcp-adapters)
6. [What is ReAct?](#6-what-is-react)
7. [MCP vs ReAct — Complementary, Not Competing](#7-mcp-vs-react--complementary-not-competing)
8. [DuckDuckGo MCP Server Architecture](#8-duckduckgo-mcp-server-architecture)
9. [Wikipedia MCP Server Architecture](#9-wikipedia-mcp-server-architecture)
10. [MultiServerMCPClient](#10-multiservermcpclient)
11. [MCP Transports: stdio vs Streamable HTTP](#11-mcp-transports-stdio-vs-streamable-http)
12. [Project Architecture & Sequence Flow](#12-project-architecture--sequence-flow)
13. [Implemented Module Architecture (`app/` & `ask.py`)](#13-implemented-module-architecture-app--askpy)
14. [Installation & macOS Setup](#14-installation--macos-setup)
15. [Environment Configuration](#15-environment-configuration)
16. [Commands to Test and Run](#16-commands-to-test-and-run)
17. [Running the Examples (01 to 06)](#17-running-the-examples-01-to-06)
18. [Safe Execution Logging](#18-safe-execution-logging)
19. [Troubleshooting & Dependency Isolation](#19-troubleshooting--dependency-isolation)
20. [Final Learning Checklist](#20-final-learning-checklist)
21. [What I Learned & Interview Questions](#21-what-i-learned--interview-questions)

---

## 1. What is MCP?

The **Model Context Protocol (MCP)** is an open standard created by Anthropic that standardizes how AI models and applications interact with external tools, contextual resources, and prompts.

Before MCP, integrating a tool into an AI application required custom glue code for every service:

- Custom API wrappers
- Bespoke schemas
- Framework-specific implementations

MCP solves the "N×M integration problem" by defining a universal protocol (based on JSON-RPC 2.0). If a service implements an MCP server, any MCP-compliant client (LangChain, Claude Desktop, Cursor, etc.) can discover and execute its tools without bespoke code.

---

## 2. What is an MCP Server?

An **MCP Server** is an independent process or web service that exposes:

- **Tools**: Executable functions (e.g., searching DuckDuckGo, fetching Wikipedia articles).
- **Resources**: Data or documents (e.g., file contents, API responses).
- **Prompts**: Reusable prompt templates.

Crucially, an MCP server runs as a separate process from your application. It advertises its capabilities dynamically via standardized endpoints (e.g., `tools/list`) and executes requests sent via `tools/call`.

---

## 3. What is an MCP Client?

An **MCP Client** is the software component that connects to one or more MCP servers. It handles:

1. Process lifecycle (spawning stdio subprocesses or opening HTTP connections).
2. Protocol handshakes (`initialize` negotiation).
3. Capability discovery (querying `tools/list`).
4. RPC invocation (sending `tools/call` and receiving formatted responses).

In this project, `MultiServerMCPClient` from `langchain-mcp-adapters` acts as our high-level MCP client.

---

## 4. What is an MCP Tool?

An **MCP Tool** is a capability published by an MCP server. Every MCP tool provides:

- A unique `name` (e.g., `search`, `get_summary`).
- A human-readable `description` explaining what the tool does and when to use it.
- An `inputSchema` formatted in standard JSON Schema detailing required and optional parameters with their types.

The LLM inspects this name, description, and schema to decide if and how to call the tool.

---

## 5. What is langchain-mcp-adapters?

`langchain-mcp-adapters` is the official bridge connecting the MCP ecosystem to the LangChain/LangGraph ecosystem.

```
MCP Server (stdio / HTTP)
       │
       ▼  (JSON-RPC 2.0)
langchain-mcp-adapters (MultiServerMCPClient)
       │
       ▼  (converts MCP tool schemas)
LangChain BaseTool instances
       │
       ▼
LangGraph ReAct Agent
```

It translates MCP tool definitions into standard LangChain `BaseTool` objects, handling:

- Argument serialization and JSON Schema conversion.
- Async tool dispatch (`ainvoke`) via the MCP protocol.
- Error handling and result mapping.

---

## 6. What is ReAct?

**ReAct** (**Rea**soning + **Act**ing) is an agent paradigm where the language model alternates between:

1. **Reasoning**: Analyzing the query, state, and previous observations.
2. **Acting**: Selecting a tool and producing structured arguments.
3. **Observing**: Receiving the tool result from the environment and continuing the cycle until the task is complete.

In modern LangChain, ReAct is built using state graphs (`langchain.agents.create_agent` or `langgraph.prebuilt.create_react_agent`), where a model node and a tools node loop until no further tool calls are emitted.

---

## 7. MCP vs ReAct — Complementary, Not Competing

A frequent misconception is that MCP and ReAct are competing technologies. They operate at completely different layers of the AI stack:

| Concept   | Question Answered                        | Role                          | Analogy                                     |
| :-------- | :--------------------------------------- | :---------------------------- | :------------------------------------------ |
| **MCP**   | _How does an application talk to tools?_ | **Communication Protocol**    | USB-C cable & drivers                       |
| **ReAct** | _How does the agent decide what to do?_  | **Decision-Making Algorithm** | The person deciding when to plug in a drive |

```
                 ReAct Agent (Decision Maker)
                      │
              decides what to do
                      │
                      ▼
              MCP Tool (Capability)
                      │
              MCP Protocol (JSON-RPC)
                      │
                      ▼
              MCP Server (Subprocess / Service)
                      │
                      ▼
             External Service (DuckDuckGo / Wikipedia)
```

- ReAct can work with standard Python functions without MCP.
- MCP can be used without ReAct (e.g., direct UI tool calling).
- **ReAct + MCP** combines flexible LLM reasoning with standard, cross-platform tool connectivity.

---

## 8. DuckDuckGo MCP Server Architecture

This project connects to the open-source `duckduckgo-mcp-server` package via `stdio`.

- **Transport**: `stdio` (runs locally as an isolated subprocess).
- **Exposed Tools**:
  - `search`: Queries DuckDuckGo for web pages, news, and links.
  - `fetch_content`: Extracts clean text or markdown content from specific web URLs.
  - `expand_link`: Resolves shortened reference tokens (`ref://...`) into full URLs.
- **Ideal For**: Real-time queries, current events, and live web exploration.

---

## 9. Wikipedia MCP Server Architecture

This project connects to the open-source `wikipedia-mcp` package via `stdio`.

- **Transport**: `stdio` (subprocess communication over stdin/stdout).
- **Exposed Tools**:
  - `get_summary`: Fetches authoritative introductory summaries of Wikipedia articles.
  - `search_wikipedia`: Searches article titles and snippets matching a query.
  - `get_article`: Retrieves complete article text and structure.
  - `extract_key_facts`: Extracts structured key facts from a given subject.
- **Ideal For**: Timeless encyclopedic knowledge, biographical history, and scientific background.

---

## 10. MultiServerMCPClient

When an agent needs multiple distinct capabilities (e.g., encyclopedia facts _and_ live web searches), `MultiServerMCPClient` aggregates tools from several independent MCP servers:

```
                         ReAct Agent
                              │
                              ▼
                    MultiServerMCPClient
                         /          \
                        /            \
                       ▼              ▼
               DuckDuckGo MCP    Wikipedia MCP
                  (stdio)           (stdio)
                     │                 │
                     ▼                 ▼
                Search Tools     Wikipedia Tools
```

You pass a configuration dictionary specifying each server's transport, executable, and arguments. Calling `await client.get_tools()` queries all servers concurrently and returns a unified list of LangChain tools.

---

## 11. MCP Transports: stdio vs Streamable HTTP

MCP defines standardized transport mechanisms for sending JSON-RPC messages between client and server:

### 1. `stdio` (Standard Input / Output)

- **How it works**: The client spawns the MCP server as a local child subprocess and writes JSON-RPC messages directly to `stdin` and reads responses from `stdout`.
- **Best for**:
  - Local CLI or package-based tool servers.
  - Simple development without network overhead.
  - Environments where opening listening ports is undesirable or restricted.

### 2. `Streamable HTTP`

- **How it works**: Modern MCP network transport operating over HTTP POST requests with streaming chunked responses.
- **Best for**:
  - Remote MCP servers hosted on another machine or container.
  - Multi-tenant server deployments.
  - Network-accessible microservices.

### 3. `HTTP + SSE` (Legacy)

- **How it works**: Uses Server-Sent Events (SSE) for server-to-client streaming and separate HTTP POST requests for client-to-server requests.
- **Status**: Retained for backward compatibility with older MCP implementations, but superseded by Streamable HTTP in modern specifications.

---

## 12. Project Architecture & Sequence Flow

### Single-Server Flow

```
User Query
    ↓
ReAct Agent
    ↓
LLM
    ↓
Does query require external data? ──[NO]──> Direct Answer
    │
  [YES]
    ↓
Select MCP Tool
    ↓
MCP Client
    ↓ (stdio)
MCP Server
    ↓
External Service
    ↓
Tool Result
    ↓
ReAct Agent
    ↓
Final Answer
```

### Multi-Server Collaborative Flow

```
                         User Query
                             │
                             ▼
                        ReAct Agent
                             │
                             ▼
                            LLM
                             │
                             ▼
                   MultiServerMCPClient
                       /           \
                      /             \
                     ▼               ▼
           DuckDuckGo MCP       Wikipedia MCP
              (stdio)              (stdio)
                 │                    │
                 ▼                    ▼
          Search/Web data       Wikipedia data
                 \                    /
                  \                  /
                   ▼                ▼
                     Tool Results
                          │
                          ▼
                     ReAct Agent
                          │
                          ▼
                     Final Answer
```

---

## 13. Implemented Module Architecture (`app/` & `ask.py`)

The project is structured into clean, modular components designed for both educational clarity and production-grade reliability:

```
mcp-react-duckduckgo-wikipedia/
│
├── app/
│   ├── __init__.py               # Package exports
│   ├── config.py                 # Dynamic runner resolution & multi-provider LLM factory
│   ├── mcp_client.py             # stdio MCP connection managers (MultiServerMCPClient)
│   └── react_agent.py            # LangChain 1.4+ ReAct agent builder & safe logging
│
├── ask.py                        # Dedicated CLI entrypoint for asking questions (One-shot & REPL)
│
├── examples/
│   ├── 01_connect_duckduckgo.py  # DuckDuckGo MCP connection over stdio
│   ├── 02_connect_wikipedia.py   # Wikipedia MCP connection over stdio
│   ├── 03_list_mcp_tools.py      # Programmatic tool discovery & JSON Schema inspection
│   ├── 04_react_duckduckgo.py    # ReAct agent deciding when to use DuckDuckGo
│   ├── 05_react_wikipedia.py     # ReAct agent querying Wikipedia facts
│   └── 06_multi_server_agent.py  # Multi-server agent combining Wikipedia + DuckDuckGo
│
├── tests/
│   ├── __init__.py
│   └── test_configuration.py     # Automated unit test suite (7 tests)
│
├── requirements.txt              # Pinned, tested dependencies
├── .env.example                  # Environment configuration template
└── README.md                     # Documentation & guide
```

### Component Breakdown:

1. **`app/config.py`**:
   - **Dynamic Runner Resolution (`resolve_uvx_path`)**: Automatically searches for the `uvx` executable in the environment, system PATH, or virtual environment (`.venv/bin/uvx`), with fallback to `UVX_COMMAND`.
   - **Multi-Provider LLM Factory (`get_chat_model`)**: Seamlessly instantiates models for **Groq** (`ChatGroq`), **Gemini** (`ChatGoogleGenerativeAI`), **Mistral** (`ChatMistralAI`), **OpenAI** (`ChatOpenAI`), and a zero-key **Demo** (`EducationalDemoChatModel`).
   - **Tool Content Formatter (`_clean_and_format_tool_content`)**: Unwraps raw JSON payloads from MCP tools (e.g. Wikipedia summaries) into structured markdown overviews.

2. **`app/mcp_client.py`**:
   - **Process Isolation via `uvx`**: Spawns `duckduckgo-mcp-server` and `wikipedia-mcp` in isolated environments over `stdio`, avoiding package conflicts between LangChain and MCP server dependencies.
   - **Warning & Log Cleanliness**: Injects `PYTHONWARNINGS="ignore"` and `--log-level ERROR` into subprocess execution environments, silencing upstream MCP deprecation warnings (`SEP-2577`).
   - **`MultiServerMCPClient`**: Uses `langchain-mcp-adapters` to query both servers concurrently and expose their tools as standard LangChain `BaseTool` instances.

3. **`app/react_agent.py`**:
   - **LangChain 1.4+ ReAct Builder (`build_react_agent`)**: Wraps `create_agent` with customized system prompts instructing the model on tool routing and discouraging premature chain-of-thought scratchpad tokens.
   - **Safe Streaming Logger (`run_agent_with_safe_logging`)**: Streams execution updates in real time using `agent.astream(..., stream_mode="updates")`. It logs explicit steps (`Selected tool`, `Arguments`, `Tool result`, `Final answer`) without printing private/hidden reasoning tokens.

4. **`ask.py`**:
   - **Interactive & One-Shot CLI Interface**: Direct entrypoint for users to ask questions. Automatically detects whether a query was provided as a CLI argument or opens an interactive prompt loop.

---

## 14. Installation & macOS Setup

Run the following commands in your terminal on macOS:

```bash
# 1. Navigate to the project directory
cd mcp-react-duckduckgo-wikipedia

# 2. Create a Python 3.11+ virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Verify Python version (requires Python 3.11+)
python3 --version

# 5. Install project dependencies
pip install -r requirements.txt
```

---

## 15. Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to configure your chosen LLM provider:

```dotenv
# Provider choices: "groq", "gemini", "mistral", "openai", or "demo"
LLM_PROVIDER=groq

# Recommended models:
# - Groq: "openai/gpt-oss-120b" (generous 250K TPM limit, reliable for web search)
# - Gemini: "gemini-3.5-flash"
# - Mistral: "mistral-small-latest"
# - OpenAI: "gpt-4o-mini"
LLM_MODEL=openai/gpt-oss-120b

# Temperature (0.0 for deterministic tool usage)
LLM_TEMPERATURE=0.0

# Set the API key for your chosen provider:
GROQ_API_KEY=gsk_your_groq_key_here
```

### Why Do We Need LLM API Keys for an MCP Project?

It is crucial to understand the separation between **The Tools** and **The Brain**:

- **The MCP Servers (Tools)**: DuckDuckGo and Wikipedia run locally via `uvx` and query public web endpoints. **They require 0 API keys.**
- **The LLM (Brain)**: The MCP servers cannot "think" or understand human intent. An LLM (via Groq, Gemini, Mistral, or OpenAI) is required to evaluate your question, decide _which_ MCP tool to invoke, formulate arguments, and synthesize the final answer. The API key is for this decision-making brain.

> [!TIP]
> **Zero-Key Offline Testing**:
> If you don't have an API key or want to test completely offline, set:
>
> ```dotenv
> LLM_PROVIDER=demo
> ```
>
> This activates our built-in `EducationalDemoChatModel` which tests the entire ReAct tool-calling loop locally without making external API calls!

---

## 16. Commands to Test and Run

Here is the complete command reference to run and verify the implemented module:

### 1. Interactive REPL Mode (`ask.py`)

Run the interactive CLI loop where you can ask back-to-back questions:

```bash
python ask.py
```

**Example Session**:

```text
Connecting to MCP servers (DuckDuckGo + Wikipedia)...
Connected! Loaded 25 tools.

======================================================================
🤖 MCP ReAct Assistant (DuckDuckGo + Wikipedia)
   Provider: GROQ
======================================================================
Type your question below (or type 'quit' or 'exit' to stop):

Ask a question > Who was Alan Turing?
... (Agent autonomously selects Wikipedia 'get_summary' tool)

Ask a question > quit
Goodbye!
```

### 2. One-Shot Question Mode (`ask.py`)

Pass any question directly as a command-line argument for a fast, one-off answer:

```bash
# Biographical / Encyclopedia search (routes to Wikipedia):
python ask.py "Who is Sachin Tendulkar?"

# General timeless question (answered directly from LLM knowledge):
python ask.py "What is Python?"
```

### 3. Automated Test Suite (`pytest`)

Run the test suite to verify configuration parsing, dynamic runner resolution, multi-provider model creation, and safe execution logging:

```bash
# Run all tests:
pytest

# Run with detailed verbose output:
pytest tests/test_configuration.py -v
```

### 4. Zero-Key Offline Verification

To test the complete ReAct tool routing loop without requiring an API key or internet quota:

```bash
# Set provider to demo:
export LLM_PROVIDER=demo

# Run with any query:
python ask.py "Who was Alan Turing?"
```

---

## 17. Running the Examples (01 to 06)

All examples are self-contained and can be run independently:

### Step 1: Verify DuckDuckGo MCP Server Connection

```bash
python examples/01_connect_duckduckgo.py
```

_Expected Output_: Connects via stdio and discovers `search`, `fetch_content`, and `expand_link`.

### Step 2: Verify Wikipedia MCP Server Connection

```bash
python examples/02_connect_wikipedia.py
```

_Expected Output_: Connects via stdio and discovers 22 Wikipedia tools (including `get_summary`, `search_wikipedia`).

### Step 3: Programmatic Tool Discovery & Schema Inspection

```bash
python examples/03_list_mcp_tools.py
```

_Expected Output_: Programmatically iterates over all discovered tools from both servers and prints their names, descriptions, and JSON input schemas without hardcoding.

### Step 4: Single-Server ReAct Agent (DuckDuckGo)

```bash
python examples/04_react_duckduckgo.py
```

_Expected Behavior_:

- Query 1 ("What is Python?"): Answered directly from model knowledge without calling DuckDuckGo.
- Query 2 ("What are the latest developments in Python?"): Triggers DuckDuckGo search tool call.

### Step 5: Single-Server ReAct Agent (Wikipedia)

```bash
python examples/05_react_wikipedia.py
```

_Expected Behavior_:

- Answers queries about Alan Turing and the history of Python using live Wikipedia MCP tools.

### Step 6: Multi-Server ReAct Agent (DuckDuckGo + Wikipedia)

```bash
python examples/06_multi_server_agent.py
```

_Expected Behavior_:

- Solves a multi-part query:
  _"Find information about Alan Turing and give me a short summary using reliable information from Wikipedia, then search the web for any recent developments related to Alan Turing."_
- Autonomously selects Wikipedia for historical background and DuckDuckGo for recent news, without hardcoding the tool order.

---

### How to Ask Custom Questions

You have 3 convenient ways to ask your own questions:

#### Method 1: Pass Question as a Command-Line Argument

Pass any question in quotes directly after the script name:

```bash
# Ask the multi-server agent:
python examples/06_multi_server_agent.py "Who was Ada Lovelace?"

# Ask for recent news:
python examples/06_multi_server_agent.py "What are the latest updates on quantum computing?"

# Ask DuckDuckGo agent directly:
python examples/04_react_duckduckgo.py "Latest news about artificial intelligence"

# Ask Wikipedia agent directly:
python examples/05_react_wikipedia.py "Explain the history of the Linux kernel"
```

#### Method 2: Interactive Terminal REPL Mode

Run with `--interactive` (or `-i`) to start a live conversational prompt:

```bash
python examples/06_multi_server_agent.py --interactive
```

Output:

```text
Ask a question > Who was Nikola Tesla?
... (runs agent and displays safe execution trace)
Ask a question > quit
```

#### Method 3: In Python Code

If you are writing your own script or integrating into an application:

```python
import asyncio
from app.config import get_chat_model
from app.mcp_client import create_multi_server_client
from app.react_agent import build_react_agent, run_agent_with_safe_logging

async def ask(my_question: str):
    # 1. Initialize model and tools
    model = get_chat_model()
    client = create_multi_server_client()
    tools = await client.get_tools()

    # 2. Build the ReAct agent
    agent = build_react_agent(model, tools)

    # 3. Ask your question!
    await run_agent_with_safe_logging(agent, my_question)

asyncio.run(ask("What are the major discoveries of the James Webb Space Telescope?"))
```

### Running Automated Tests

```bash
pytest tests/test_configuration.py
```

---

## 18. Safe Execution Logging

In accordance with responsible AI observability best practices, this project does **NOT** print internal chain-of-thought tokens or hidden reasoning steps (e.g. `Thought: I should search...`).

Instead, clean and structured execution traces are logged:

```text
User:
Who was Alan Turing?

Selected tool:
get_summary

Arguments:
title="Alan Turing"

Tool result:
{"title":"Alan Turing","summary":"Alan Mathison Turing was an English mathematician..."}

Final answer:
Alan Turing (1912–1954) was an English mathematician and pioneer of computer science...
```

---

## 19. Troubleshooting & Dependency Isolation

### Why Use `uvx` for MCP Servers?

During integration, we observed that `langchain-mcp-adapters 0.3.2` requires `mcp<2.0.0,>=1.24.0`, while certain modern standalone Python MCP packages (such as `duckduckgo-mcp-server`) require `mcp>=2.1.1`.

Attempting to install both into a single Python environment causes dependency conflicts.

**The Solution:**
Because MCP is inherently designed around process isolation, our `app/mcp_client.py` uses `uvx` to launch the MCP servers in their own isolated, ephemeral environments. The host application runs cleanly with `langchain-mcp-adapters`, while the MCP servers execute independently.

---

## 20. Final Learning Checklist

Verify your understanding by checking off these core concepts:

- [x] **What MCP is**: A standardized JSON-RPC protocol enabling AI clients to connect to any tool server.
- [x] **What an MCP server is**: An independent service that advertises tools, resources, and prompts.
- [x] **What an MCP client is**: The software managing server connections and tool invocations.
- [x] **What an MCP tool is**: A named function defined with descriptions and JSON Schema parameters.
- [x] **How tool discovery works**: Clients call `tools/list` on the server during initialization.
- [x] **What langchain-mcp-adapters does**: Translates MCP schemas into LangChain `BaseTool` objects.
- [x] **What a ReAct agent does**: Iteratively reasons and acts using tools until reaching a final answer.
- [x] **Why tools aren't called every time**: ReAct agents treat tools as capabilities; general questions are answered directly.
- [x] **How DuckDuckGo MCP works**: Provides live web search and webpage fetching over stdio.
- [x] **How Wikipedia MCP works**: Provides factual encyclopedia article lookups and summaries over stdio.
- [x] **How MultiServerMCPClient works**: Connects to multiple MCP servers concurrently and aggregates their tools.
- [x] **stdio vs Streamable HTTP**: stdio is for local subprocesses; Streamable HTTP is for remote network services.
- [x] **Difference between MCP and ReAct**: MCP is how to _communicate_ with tools; ReAct is how to _decide_ when to use them.
- [x] **Complete Request Flow**: User → ReAct Agent → LLM → MCP Tool → MCP Server → Service → Tool Result → Agent → Final Answer.

---

## 21. What I Learned & Interview Questions

### Key Takeaways

1. **Separation of Concerns**: Tools are capabilities (MCP), while agents are decision-makers (ReAct). Decoupling them allows tools to be written once in any language and reused across any agent framework.
2. **Process Isolation**: MCP servers communicate over stdio or HTTP, meaning they don't share Python dependencies with your application. This prevents package version conflicts.
3. **Adaptive Tool Usage**: A well-prompted ReAct agent does not blindly invoke tools for simple questions where internal knowledge suffices.

### Example Interview Questions

**Q1: What is the primary difference between LangChain Tools and MCP Tools?**

> _Answer_: LangChain Tools are framework-specific Python/JavaScript classes. MCP Tools are language-agnostic capabilities exposed over a JSON-RPC 2.0 protocol. MCP tools can be written in any language (Python, Node.js, Go, Rust) and consumed by any MCP client, including LangChain via `langchain-mcp-adapters`.

**Q2: When would you use stdio transport over Streamable HTTP in MCP?**

> _Answer_: Use `stdio` for local tools running on the same machine as child subprocesses (e.g. desktop assistants, local CLI utilities). Use `Streamable HTTP` when tools run on remote servers, in microservices architectures, or need to be shared across multiple distributed clients.

**Q3: How does a ReAct agent decide between multiple tools from different MCP servers?**

> _Answer_: The LLM receives the descriptions and JSON schemas of all tools in its system prompt/function definitions. When a user asks a question, the LLM evaluates the semantic suitability of each tool (e.g., choosing Wikipedia for encyclopedia facts and DuckDuckGo for recent news) and emits structured tool-call tokens.
