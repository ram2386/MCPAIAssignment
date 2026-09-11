"""
mcp-react-duckduckgo-wikipedia
Hands-on educational project for Model Context Protocol (MCP) and ReAct agents.
"""

from app.config import get_chat_model, resolve_uvx_path
from app.mcp_client import (
    create_duckduckgo_client,
    create_wikipedia_client,
    create_multi_server_client,
)
from app.react_agent import build_react_agent, run_agent_with_safe_logging

__all__ = [
    "get_chat_model",
    "resolve_uvx_path",
    "create_duckduckgo_client",
    "create_wikipedia_client",
    "create_multi_server_client",
    "build_react_agent",
    "run_agent_with_safe_logging",
]
