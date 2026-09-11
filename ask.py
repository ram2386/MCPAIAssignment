#!/usr/bin/env python3
"""
CLI Question-and-Answer Tool for mcp-react-duckduckgo-wikipedia
===============================================================

Usage:
    # 1. Ask a question directly via command-line argument:
    python ask.py "Who was Alan Turing?"
    python ask.py "What are the latest developments in Python?"
    python ask.py "Find information about Ada Lovelace on Wikipedia and search recent news"

    # 2. Interactive prompt mode (prompts you for questions in the terminal):
    python ask.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import get_chat_model
from app.mcp_client import create_multi_server_client
from app.react_agent import build_react_agent, run_agent_with_safe_logging


async def main():
    provider = os.getenv("LLM_PROVIDER", "demo").strip().lower()

    # 1. Initialize the language model
    try:
        model = get_chat_model()
    except Exception as e:
        print(f"\n[Configuration Error]\n{e}\n")
        print("Tip: Set LLM_PROVIDER=demo in .env to run without an API key.")
        return

    # 2. Connect to both MCP servers over stdio
    print("Connecting to MCP servers (DuckDuckGo + Wikipedia)...")
    client = create_multi_server_client()
    tools = await client.get_tools()
    print(f"Connected! Loaded {len(tools)} tools.\n")

    # 3. Build ReAct agent
    agent = build_react_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are a helpful research assistant with access to DuckDuckGo web search and Wikipedia.\n"
            "- For foundational or general questions, answer directly.\n"
            "- For encyclopedic, biographical, or historical background, use Wikipedia tools.\n"
            "- For recent news, current events, or web queries, use DuckDuckGo search.\n"
            "- Synthesize the answer clearly and concisely."
        ),
    )

    # 4. Check if question was passed as CLI arguments
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    if args:
        # One-shot mode: question passed on command line
        question = " ".join(args)
        await run_agent_with_safe_logging(agent, question)
    else:
        # Interactive mode: prompt the user in terminal
        print("=" * 70)
        print("🤖 MCP ReAct Assistant (DuckDuckGo + Wikipedia)")
        print(f"   Provider: {provider.upper()}")
        print("=" * 70)
        print("Type your question below (or type 'quit' or 'exit' to stop):\n")

        while True:
            try:
                question = input("Ask a question > ").strip()
                if not question:
                    continue
                if question.lower() in ("quit", "exit", "q"):
                    print("\nGoodbye!")
                    break
                await run_agent_with_safe_logging(agent, question)
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye!")
                break


if __name__ == "__main__":
    asyncio.run(main())
