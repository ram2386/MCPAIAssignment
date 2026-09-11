"""
Example 01: Connect to DuckDuckGo MCP Server
============================================

Learning Objectives:
1. Understand what an MCP server is (an external process exposing capabilities).
2. Connect to an existing open-source MCP server (duckduckgo-mcp-server) using stdio transport.
3. Understand how langchain-mcp-adapters converts MCP tools into LangChain BaseTool objects.

Architecture Flow:
    duckduckgo-mcp-server (subprocess)
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

from app.mcp_client import create_duckduckgo_client


async def main():
    print("=" * 65)
    print("STEP 1: Initializing DuckDuckGo MCP Client (MultiServerMCPClient)")
    print("=" * 65)
    print("Transport: stdio (subprocess communication over stdin/stdout)")
    print("Server:    duckduckgo-mcp-server (via isolated runner)")

    client = create_duckduckgo_client()

    print("\nConnecting to DuckDuckGo MCP server and discovering tools...")
    # client.get_tools() connects to the server, sends 'tools/list',
    # and wraps each MCP tool as a LangChain BaseTool.
    tools = await client.get_tools()

    print(f"\nSuccessfully connected! Discovered {len(tools)} tool(s):")
    for idx, tool in enumerate(tools, start=1):
        print(f"  {idx}. {tool.name}: {tool.description.splitlines()[0]}")

    print("\n" + "=" * 65)
    print("CONCEPT RECAP:")
    print("  - The MCP server runs as a separate subprocess.")
    print("  - We did NOT hardcode tool definitions in Python.")
    print("  - langchain-mcp-adapters converted them into LangChain BaseTool objects.")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
