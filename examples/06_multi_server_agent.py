r"""
Example 06: Multi-Server ReAct Agent (DuckDuckGo + Wikipedia)
============================================================

Learning Objectives:
1. Connect multiple heterogeneous MCP servers simultaneously using MultiServerMCPClient.
2. Provide all aggregated tools to a single LangChain ReAct agent without manual wrapping.
3. Observe the ReAct agent autonomously selecting Wikipedia for historical background
   and DuckDuckGo for recent news/web data.
4. Verify that the tool sequence is NOT hardcoded—the ReAct loop decides dynamically.

Multi-Server Architecture:
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
"""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path so 'app' can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_chat_model
from app.mcp_client import create_multi_server_client
from app.react_agent import build_react_agent, run_agent_with_safe_logging


async def main():
    print("=" * 70)
    print("DEMO: Multi-Server MCP ReAct Agent (DuckDuckGo + Wikipedia)")
    print("=" * 70)

    # 1. Initialize Chat Model
    try:
        model = get_chat_model()
    except Exception as e:
        print(f"\nConfiguration error:\n{e}\n")
        return

    # 2. Connect to BOTH MCP servers simultaneously using MultiServerMCPClient
    print("\nConnecting to MultiServerMCPClient (DuckDuckGo + Wikipedia)...")
    multi_client = create_multi_server_client()
    tools = await multi_client.get_tools()

    print(f"\nSuccessfully loaded {len(tools)} total tools across both servers!")
    print("DuckDuckGo tools provide: web search, webpage fetching, link expansion.")
    print("Wikipedia tools provide:  article summaries, full articles, key facts, sections.\n")

    # 3. Build ReAct agent with tools from both servers
    agent = build_react_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are a sophisticated research assistant with access to both Wikipedia and DuckDuckGo.\n"
            "- Use Wikipedia tools for biographical, historical, and encyclopedia background.\n"
            "- Use DuckDuckGo tools for current news, recent events, or broad web exploration.\n"
            "- If a query requires both background knowledge and recent developments,\n"
            "  decide autonomously which tool to call first and synthesize the final answer seamlessly."
        ),
    )

    # 4. Check for user question from CLI arguments or interactive mode
    cli_args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    interactive_mode = any(arg in ("-i", "--interactive") for arg in sys.argv[1:])

    if interactive_mode:
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE: Type your questions below (or 'quit' / 'exit' to stop)")
        print("=" * 70)
        while True:
            try:
                user_input = input("\nAsk a question > ").strip()
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
        # Default educational task
        default_query = (
            "Find information about Alan Turing and give me a short summary using reliable "
            "information from Wikipedia, then search the web for any recent developments related to Alan Turing."
        )
        print(">>> Executing Default Multi-Server Reasoning Task <<<")
        print("(Tip: You can pass your own question: python examples/06_multi_server_agent.py \"Your question\")")
        print("(Tip: Or run interactively: python examples/06_multi_server_agent.py --interactive)\n")
        await run_agent_with_safe_logging(agent, default_query)


if __name__ == "__main__":
    asyncio.run(main())
