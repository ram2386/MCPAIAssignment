"""
Example 03: Discover MCP Tools Programmatically
===============================================

Learning Objectives:
1. Understand that MCP servers dynamically advertise their capabilities via tool schemas.
2. Programmatically inspect tool names, descriptions, and input schemas without hard-coding.
3. Understand the discovery pipeline:
   MCP Server → Tool Discovery (tools/list) → Tool Definitions (JSON Schema) → LangChain Tools (BaseTool)
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to sys.path so 'app' can be imported directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.mcp_client import create_duckduckgo_client, create_wikipedia_client


async def inspect_server_tools(server_label: str, client_factory):
    """Connects to an MCP client and prints the discovered tool specifications."""
    client = client_factory()
    tools = await client.get_tools()

    for tool in tools:
        print(f"Server:\n{server_label}\n")
        print(f"Tool:\n{tool.name}\n")
        print(f"Description:\n{tool.description.strip()}\n")

        # In LangChain BaseTool, args or args_schema defines the input parameters schema
        schema_dict = tool.args if hasattr(tool, "args") else {}
        formatted_schema = json.dumps(schema_dict, indent=2)
        print(f"Input schema:\n{formatted_schema}\n")
        print("-" * 65)


async def main():
    print("=" * 65)
    print("TOOL DISCOVERY PIPELINE")
    print("  MCP Server  →  Tool Discovery  →  Tool Definitions  →  LangChain Tools")
    print("=" * 65 + "\n")

    # 1. Discover DuckDuckGo Tools
    await inspect_server_tools("DuckDuckGo", create_duckduckgo_client)

    # 2. Discover Wikipedia Tools
    await inspect_server_tools("Wikipedia", create_wikipedia_client)


if __name__ == "__main__":
    asyncio.run(main())
