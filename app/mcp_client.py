"""
MCP Client Manager for mcp-react-duckduckgo-wikipedia.

Architecture:
    MCP Server (subprocess running duckduckgo-mcp-server or wikipedia-mcp)
        ↓  (JSON-RPC 2.0 messages over stdio)
    MCP Client (underlying mcp protocol client)
        ↓
    langchain-mcp-adapters (converts MCP tools into LangChain BaseTool instances)
        ↓
    LangChain Tools (callable by ReAct Agent)

Why stdio transport?
    `stdio` launches the MCP server as an isolated child subprocess and exchanges
    JSON-RPC messages over stdin and stdout. This is the recommended transport for
    local tools: it requires no open network ports, no local firewall exceptions,
    and no authentication tokens.
"""

import os
from typing import Any, Dict
from langchain_mcp_adapters.client import MultiServerMCPClient
from app.config import resolve_uvx_path


def get_duckduckgo_connection_config() -> Dict[str, Any]:
    """
    Returns the Stdio connection configuration for the open-source DuckDuckGo MCP server.
    Uses 'uvx' to run the package in an isolated virtual environment.
    Silences internal MCP deprecation warnings via PYTHONWARNINGS=ignore.
    """
    uvx_path = resolve_uvx_path()
    return {
        "transport": "stdio",
        "command": uvx_path,
        "args": ["duckduckgo-mcp-server"],
        "env": {
            **os.environ,
            "PYTHONWARNINGS": "ignore",
        },
    }


def get_wikipedia_connection_config() -> Dict[str, Any]:
    """
    Returns the Stdio connection configuration for the open-source Wikipedia MCP server.
    Uses 'uvx' to run the package in an isolated virtual environment with quiet logging.
    """
    uvx_path = resolve_uvx_path()
    return {
        "transport": "stdio",
        "command": uvx_path,
        "args": ["wikipedia-mcp", "--log-level", "ERROR"],
        "env": {
            **os.environ,
            "PYTHONWARNINGS": "ignore",
        },
    }


def create_duckduckgo_client() -> MultiServerMCPClient:
    """
    Create a MultiServerMCPClient connected solely to the DuckDuckGo MCP server.
    """
    return MultiServerMCPClient({
        "duckduckgo": get_duckduckgo_connection_config()
    })


def create_wikipedia_client() -> MultiServerMCPClient:
    """
    Create a MultiServerMCPClient connected solely to the Wikipedia MCP server.
    """
    return MultiServerMCPClient({
        "wikipedia": get_wikipedia_connection_config()
    })


def create_multi_server_client() -> MultiServerMCPClient:
    """
    Create a MultiServerMCPClient connected simultaneously to BOTH:
    1. DuckDuckGo MCP server (current web search & fetching)
    2. Wikipedia MCP server (factual & encyclopedia articles)
    """
    return MultiServerMCPClient({
        "duckduckgo": get_duckduckgo_connection_config(),
        "wikipedia": get_wikipedia_connection_config(),
    })
