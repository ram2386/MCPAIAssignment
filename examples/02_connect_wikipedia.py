"""
Example 02: Connect to Wikipedia MCP Server
===========================================

Learning Objectives:
1. Connect to an existing open-source Wikipedia MCP server (wikipedia-mcp).
2. Understand stdio transport for local tool servers.
3. Observe how Wikipedia articles, summaries, and searches are exposed as LangChain tools.

Architecture Flow:
      wikipedia-mcp (subprocess)
            ↓  (JSON-RPC 2.0 over stdio)
       MCP Client (stdio_client)
            ↓
    langchain-mcp-adapters
            ↓
     LangChain Tools
"""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path so 'app' can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.mcp_client import create_wikipedia_client


async def main():
    print("=" * 65)
    print("STEP 1: Initializing Wikipedia MCP Client (MultiServerMCPClient)")
    print("=" * 65)
    print("Transport: stdio (subprocess communication over stdin/stdout)")
    print("Server:    wikipedia-mcp (via isolated runner)")

    client = create_wikipedia_client()

    print("\nConnecting to Wikipedia MCP server and discovering tools...")
    # client.get_tools() connects to the server, queries its tool capabilities,
    # and generates LangChain BaseTool instances dynamically.
    tools = await client.get_tools()

    print(f"\nSuccessfully connected! Discovered {len(tools)} tool(s):")
    for idx, tool in enumerate(tools, start=1):
        first_line = tool.description.splitlines()[0] if tool.description else "No description"
        print(f"  {idx:2d}. {tool.name:<30} : {first_line}")

    print("\n" + "=" * 65)
    print("CONCEPT RECAP:")
    print("  - The Wikipedia server advertises factual lookup capabilities.")
    print("  - Tools like get_summary, search_wikipedia, and get_article")
    print("    are ready for any LangChain agent or chain.")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
