"""
Unit tests for configuration, connection configs, and agent graph compilation.
These tests run in isolation and do not require external LLM API keys.
"""

import os
import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from app.config import resolve_uvx_path, get_chat_model
from app.mcp_client import (
    get_duckduckgo_connection_config,
    get_wikipedia_connection_config,
)
from app.react_agent import build_react_agent, run_agent_with_safe_logging


def test_resolve_uvx_path():
    """Verify that uvx path can be resolved and points to an existing binary."""
    path = resolve_uvx_path()
    assert isinstance(path, str)
    assert os.path.exists(path), f"Resolved uvx path does not exist: {path}"


def test_missing_api_key_raises_error(monkeypatch):
    """Verify that missing or placeholder API key raises a clear ValueError."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ValueError) as excinfo:
        get_chat_model()
    assert "Missing GROQ_API_KEY" in str(excinfo.value)


def test_invalid_provider_raises_error(monkeypatch):
    """Verify that an unsupported provider name raises an informative ValueError."""
    monkeypatch.setenv("LLM_PROVIDER", "unsupported_llm_xyz")
    with pytest.raises(ValueError) as excinfo:
        get_chat_model()
    assert "Unsupported LLM_PROVIDER" in str(excinfo.value)


def test_duckduckgo_connection_config():
    """Verify DuckDuckGo MCP connection configuration structure."""
    config = get_duckduckgo_connection_config()
    assert config["transport"] == "stdio"
    assert "command" in config
    assert config["args"] == ["duckduckgo-mcp-server"]


def test_wikipedia_connection_config():
    """Verify Wikipedia MCP connection configuration structure."""
    config = get_wikipedia_connection_config()
    assert config["transport"] == "stdio"
    assert "command" in config
    assert "wikipedia-mcp" in config["args"]


@tool
def sample_calculator(expression: str) -> str:
    """Evaluate math expressions."""
    return str(eval(expression))


def test_build_react_agent():
    """Verify that build_react_agent compiles a runnable LangGraph StateGraph."""
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model

    agent = build_react_agent(mock_model, [sample_calculator])
    assert agent is not None
    # Verify graph has the expected compiled nodes
    assert hasattr(agent, "astream")
    assert hasattr(agent, "ainvoke")


@pytest.mark.asyncio
async def test_run_agent_with_safe_logging():
    """Verify that safe execution logging streams tool calls and answers properly."""
    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model

    # Simulate 2 steps: first step makes tool call, second step produces final answer
    step1 = AIMessage(
        content="",
        tool_calls=[{
            "name": "sample_calculator",
            "args": {"expression": "2 + 2"},
            "id": "call_1",
            "type": "tool_call",
        }],
    )
    step2 = AIMessage(content="The result of 2 + 2 is 4.")

    async def mock_ainvoke(messages, **kwargs):
        # Initial turn has 1 user message (or system + user)
        if len(messages) <= 2:
            return step1
        return step2

    mock_model.ainvoke.side_effect = mock_ainvoke

    agent = build_react_agent(mock_model, [sample_calculator])
    answer = await run_agent_with_safe_logging(agent, "Calculate 2 + 2")

    assert answer == "The result of 2 + 2 is 4."
