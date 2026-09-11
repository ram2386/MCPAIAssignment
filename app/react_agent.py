"""
ReAct Agent Module for mcp-react-duckduckgo-wikipedia.

Core Concepts:
    MCP Tool    = Capability     (Standardized protocol for communicating with external services)
    ReAct Agent = Decision Maker (Agent pattern that iteratively reasons and acts)

    MCP provides the connection to tools; ReAct determines WHEN and WHICH tool to call.
    The agent does NOT blindly invoke tools for every query—if internal LLM knowledge is
    sufficient, it can respond directly without triggering an MCP round-trip.

Safe Execution Logging:
    Per design requirements, internal/hidden chain-of-thought tokens are NOT printed.
    Only explicit execution steps (tool selections, arguments, tool summaries, and final answers)
    are logged for clear observability.
"""

from typing import Any, List, Optional, Sequence
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, accurate AI research assistant equipped with MCP tools.\n"
    "Guidelines:\n"
    "1. When answering general/timeless questions (e.g. 'What is Python?'), you may answer directly from your knowledge without calling tools.\n"
    "2. When answering questions requiring verified historical, biographical, or encyclopedia knowledge, use the Wikipedia tools.\n"
    "3. When answering questions about recent developments, current events, or web search, use the DuckDuckGo tools.\n"
    "4. Do NOT output scratchpad thoughts or conversational preamble before calling a tool. Invoke tools directly.\n"
    "5. Synthesize tool results cleanly, provide a direct and thorough answer, and cite sources where relevant."
)


def build_react_agent(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: Optional[str] = None,
):
    """
    Build a ReAct agent using the current LangChain 1.4+ create_agent API.

    Args:
        model: BaseChatModel instance (e.g. ChatGroq, ChatGoogleGenerativeAI, ChatMistralAI, ChatOpenAI)
        tools: Sequence of LangChain BaseTool objects loaded from MCP servers
        system_prompt: Optional system prompt guiding the agent's tool-selection behavior

    Returns:
        CompiledStateGraph representing the ReAct agent loop.
    """
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    return create_agent(
        model=model,
        tools=list(tools),
        system_prompt=prompt,
    )


async def run_agent_with_safe_logging(
    agent: Any,
    query: str,
    max_result_length: int = 350,
) -> Optional[str]:
    """
    Execute the ReAct agent on a user query and stream safe execution logs.

    Safe Execution Trace format:
        User:
        <query>

        Selected tool:
        <tool_name>

        Arguments:
        <arg_key>=<arg_value>

        Tool result:
        <short result>

        Final answer:
        <final response>
    """
    print("=" * 70, flush=True)
    print("User:", flush=True)
    print(query, flush=True)
    print("=" * 70, flush=True)

    inputs = {"messages": [{"role": "user", "content": query}]}
    final_answer: Optional[str] = None
    tools_called = False

    try:
        async for chunk in agent.astream(inputs, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                messages = node_output.get("messages", [])
                for msg in messages:
                    # 1. Agent decision to invoke a tool
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tools_called = True
                        for tc in msg.tool_calls:
                            name = tc.get("name", "unknown_tool")
                            args = tc.get("args", {})
                            print(f"\nSelected tool:\n{name}\n", flush=True)
                            if isinstance(args, dict):
                                formatted_args = ", ".join(
                                    f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                                    for k, v in args.items()
                                )
                            else:
                                formatted_args = str(args)
                            print(f"Arguments:\n{formatted_args}\n", flush=True)

                    # 2. Tool execution result returned from MCP server
                    elif getattr(msg, "type", "") == "tool" or msg.__class__.__name__ == "ToolMessage":
                        raw = str(msg.content)
                        if len(raw) > max_result_length:
                            preview = raw[:max_result_length].strip() + "... [truncated for display]"
                        else:
                            preview = raw.strip()
                        print(f"Tool result:\n{preview}\n", flush=True)

                    # 3. Final model response (AIMessage with no new tool calls)
                    elif hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                        final_answer = str(msg.content)

    except Exception as exc:
        print(f"\n[Agent Execution Error]: {exc}", flush=True)
        if "output_parse_failed" in str(exc) or "400" in str(exc):
            print(
                "\n[Tip]: The current model returned free-form text during tool parsing.\n"
                "Switch to a dedicated tool-calling model such as 'qwen/qwen3.8-27b' in .env:\n"
                "  LLM_PROVIDER=groq\n"
                "  LLM_MODEL=qwen/qwen3.8-27b\n",
                flush=True,
            )
        raise exc

    if not tools_called:
        print("\n(Agent decided no external tool was necessary; answered directly from internal knowledge)\n", flush=True)

    if final_answer:
        print("Final answer:", flush=True)
        print(final_answer.strip(), flush=True)
        print("=" * 70 + "\n", flush=True)

    return final_answer
