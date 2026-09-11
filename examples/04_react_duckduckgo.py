"""
Example 04: Single-Server ReAct Agent (DuckDuckGo MCP)
======================================================

Learning Objectives:
1. Build a LangChain ReAct agent equipped with DuckDuckGo MCP tools.
2. Master the core ReAct principle: Tools are CAPABILITIES, not MANDATES.
   - Query 1: "What is Python?" -> Answered directly from model knowledge (NO tool call).
   - Query 2: "What are the latest developments in Python?" -> Requires recent info, triggers DuckDuckGo search.
3. Observe safe execution logging without exposing private reasoning/chain-of-thought.

ReAct Decision Flow:
    User Query
        ↓
    ReAct Agent
        ↓
       LLM
        ↓
    Does query require external/recent info?
      ├── NO  ──> Direct LLM Answer
      └── YES ──> Call DuckDuckGo MCP Tool
                       ↓
                  MCP Server
                       ↓
                  Tool Result
                       ↓
                  Final Answer
"""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path so 'app' can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_chat_model
from app.mcp_client import create_duckduckgo_client
from app.react_agent import build_react_agent, run_agent_with_safe_logging


async def main():
    print("=" * 70)
    print("DEMO: DuckDuckGo MCP ReAct Agent")
    print("=" * 70)

    # 1. Initialize Chat Model
    try:
        model = get_chat_model()
    except Exception as e:
        print(f"\nConfiguration error:\n{e}\n")
        return

    # 2. Connect to DuckDuckGo MCP server and retrieve tools
    print("\nConnecting to DuckDuckGo MCP Server...")
    client = create_duckduckgo_client()
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tool(s) from DuckDuckGo MCP server.\n")

    # 3. Build ReAct agent
    agent = build_react_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are an AI research assistant equipped with DuckDuckGo web search tools.\n"
            "If a question asks for general, well-known foundational concepts (like 'What is Python?'),\n"
            "answer directly from your knowledge without calling any tools.\n"
            "Only invoke DuckDuckGo tools when asked for current events, news, or recent developments."
        ),
    )

    # 4. Check for user question from CLI arguments or interactive mode
    cli_args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    interactive_mode = any(arg in ("-i", "--interactive") for arg in sys.argv[1:])

    if interactive_mode:
        print("\n" + "=" * 70)
        print("INTERACTIVE DUCKDUCKGO MODE (type 'quit' to exit)")
        print("=" * 70)
        while True:
            try:
                user_input = input("\nAsk DuckDuckGo Agent > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break
                await run_agent_with_safe_logging(agent, user_input)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive mode.")
                break

    elif cli_args:
        custom_query = " ".join(cli_args)
        print(f">>> Executing Custom User Query <<<\n")
        await run_agent_with_safe_logging(agent, custom_query)

    else:
        # Test Query 1: Foundational knowledge (Expects NO tool call)
        print("\n>>> TEST 1: Query requiring internal knowledge only <<<")
        query_1 = "What is Python?"
        await run_agent_with_safe_logging(agent, query_1)

        # Test Query 2: Current/external information (Expects tool call)
        print("\n>>> TEST 2: Query requiring current external search <<<")
        query_2 = "What are the latest developments in Python?"
        await run_agent_with_safe_logging(agent, query_2)


if __name__ == "__main__":
    asyncio.run(main())
