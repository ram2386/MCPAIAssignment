"""
Example 05: Single-Server ReAct Agent (Wikipedia MCP)
=====================================================

Learning Objectives:
1. Build a LangChain ReAct agent equipped with Wikipedia MCP tools.
2. Observe how the ReAct agent queries Wikipedia for biographical and historical facts.
3. Trace the execution flow:
   User Query → ReAct Agent → LLM → Wikipedia MCP Tool → MCP Server → Wikipedia Result → Agent → Final Answer.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path so 'app' can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_chat_model
from app.mcp_client import create_wikipedia_client
from app.react_agent import build_react_agent, run_agent_with_safe_logging


async def main():
    print("=" * 70)
    print("DEMO: Wikipedia MCP ReAct Agent")
    print("=" * 70)

    # 1. Initialize Chat Model
    try:
        model = get_chat_model()
    except Exception as e:
        print(f"\nConfiguration error:\n{e}\n")
        return

    # 2. Connect to Wikipedia MCP server and retrieve tools
    print("\nConnecting to Wikipedia MCP Server...")
    client = create_wikipedia_client()
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tool(s) from Wikipedia MCP server.\n")

    # 3. Build ReAct agent
    agent = build_react_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are a factual research assistant equipped with Wikipedia MCP tools.\n"
            "When asked biographical or historical questions, use the Wikipedia tools\n"
            "(such as get_summary, search_wikipedia, or get_article) to retrieve reliable encyclopedia facts.\n"
            "Provide clear, concise answers based on the retrieved data."
        ),
    )

    # 4. Check for user question from CLI arguments or interactive mode
    cli_args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    interactive_mode = any(arg in ("-i", "--interactive") for arg in sys.argv[1:])

    if interactive_mode:
        print("\n" + "=" * 70)
        print("INTERACTIVE WIKIPEDIA MODE (type 'quit' to exit)")
        print("=" * 70)
        while True:
            try:
                user_input = input("\nAsk Wikipedia Agent > ").strip()
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
        # Test Query 1: Historical figure
        print("\n>>> TEST 1: Biographical inquiry (Alan Turing) <<<")
        await run_agent_with_safe_logging(agent, "Who was Alan Turing?")

        # Test Query 2: Historical development
        print("\n>>> TEST 2: History inquiry (Python programming language) <<<")
        await run_agent_with_safe_logging(agent, "Explain the history of the Python programming language.")


if __name__ == "__main__":
    asyncio.run(main())
